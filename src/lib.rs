//! SCRFD det_10g face detection on packed RGB images. Boxes and landmarks
//! are returned in original-image coordinates.
mod cnn;
pub mod detection;
pub mod hub;
mod model;
mod onnx;
mod plan;
pub mod postprocess;
use anyhow::{Result, ensure};
pub use detection::{Detection, DetectionOptions, Image, Ranking};
use hrx::{
    execution::{Graph, MemoryPlacement},
    image::{ImageOps, RgbResize},
    inference::{Inference, InferenceGraph, ModelContext, PreparedModel},
    plan_cache::PlanCache,
    tensor::{DType, DeviceTensor, Layout, TensorDesc},
};
use std::path::Path;
pub const SIZE: usize = 640;
#[derive(Clone, Copy, Debug)]
pub struct Options {
    pub device: i32,
    pub max_batch: usize,
}
impl Default for Options {
    fn default() -> Self {
        Self {
            device: 0,
            max_batch: 16,
        }
    }
}
/// Shared resident detector with bounded private inference slots.
pub struct Scrfd {
    cnn: cnn::Cnn,
    postprocess: postprocess::Postprocess,
    images: ImageOps,
    image_plans: PlanCache<(Vec<[usize; 2]>, u32, usize), PreparedModel>,
    boxed_plans: PlanCache<(usize, u32, usize), PreparedModel>,
}
impl Scrfd {
    /// Load the pinned pretrained model from the Hugging Face cache, fetching it
    /// if needed. Set `HF_HUB_OFFLINE=1` for cached weights only.
    /// Use [`Self::load`] to supply a local file instead.
    pub fn from_pretrained(options: Options) -> Result<Self> {
        ensure!(
            (1..=64).contains(&options.max_batch),
            "max_batch must be 1..=64"
        );
        Self::load(hub::weights(false)?, options)
    }

    /// Validate and pack the model, compile kernels, and allocate resident storage.
    pub fn load(path: impl AsRef<Path>, options: Options) -> Result<Self> {
        let context = ModelContext::new(hrx::execution::RuntimeOptions {
            gpu_index: options.device,
            ..Default::default()
        })?;
        Self::load_in(path, &context, options.max_batch)
    }
    /// Load into a context shared with other models and device image operations.
    pub fn load_in(
        path: impl AsRef<Path>,
        context: &ModelContext,
        max_batch: usize,
    ) -> Result<Self> {
        ensure!((1..=64).contains(&max_batch), "max_batch must be 1..=64");
        Ok(Self {
            images: ImageOps::new(context, 8)?,
            image_plans: PlanCache::new(8, PreparedModel::is_idle)?,
            boxed_plans: PlanCache::new(8, PreparedModel::is_idle)?,
            cnn: cnn::Cnn::new(model::load(path.as_ref())?, context, max_batch)?,
            postprocess: postprocess::Postprocess::new(context)?,
        })
    }
    /// Model's explicit scheduling and allocation domain.
    pub fn context(&self) -> &ModelContext {
        self.cnn.engine.context()
    }
    /// Letterbox a resident RGB image/batch, retaining pixels on device. The
    /// returned scale matches SCRFD's original-image coordinate convention.
    pub fn letterbox(&self, image: &DeviceTensor) -> Result<(Inference, f32)> {
        self.context().validate(image)?;
        let shape = image.desc().shape();
        ensure!(shape.len() == 4, "letterbox requires NHWC RGB");
        let resize = RgbResize::letterbox(shape[1], shape[2], SIZE, SIZE, false)?;
        Ok((
            self.images.resize_rgb(image, resize)?,
            resize.region[3] as f32 / shape[1] as f32,
        ))
    }
    /// Submit resident, top-left-letterboxed NHWC RGB canvases. Returns the three
    /// padded f16 detection heads without downloading them.
    pub fn submit_heads(&self, canvases: &DeviceTensor) -> Result<Inference> {
        self.context().validate(canvases)?;
        let desc = canvases.desc();
        ensure!(
            desc.dtype() == DType::U8
                && desc.layout() == Layout::Nhwc
                && desc.is_contiguous()
                && desc.shape().len() == 4
                && desc.shape()[1..] == [SIZE, SIZE, 3],
            "expected contiguous NHWC uint8 640×640 canvases"
        );
        Ok(self
            .cnn
            .prepare(desc.shape()[0])?
            .submit(std::slice::from_ref(canvases))?)
    }

    /// Record the detector CNN into a caller-owned graph, binding canvases
    /// directly and returning three resident head tensors. Outputs become valid
    /// after that graph executes; no private pool or ingress copy is inserted.
    pub fn record_heads(
        &self,
        graph: &mut Graph,
        canvases: &DeviceTensor,
    ) -> Result<Vec<DeviceTensor>> {
        self.context().validate(canvases)?;
        let batch = canvases.desc().shape().first().copied().unwrap_or(0);
        Ok(self
            .cnn
            .fragment(batch)?
            .record(graph, std::slice::from_ref(canvases))?)
    }
    /// Record resident RGB letterboxing, CNN and candidate decoding together.
    /// The image is bound directly, without private inference-slot copies.
    /// CPU NMS is a separate explicit boundary on the returned bindings.
    pub fn record_image(
        &self,
        graph: &mut Graph,
        image: &DeviceTensor,
        options: DetectionOptions,
    ) -> Result<postprocess::RecordedDetections> {
        self.context().validate(image)?;
        options.validate()?;
        let shape = image.desc().shape();
        ensure!(
            shape.len() == 4 && (1..=self.cnn.max_batch).contains(&shape[0]),
            "invalid image batch"
        );
        let shapes = vec![[shape[2], shape[1]]; shape[0]];
        let geometry = RgbResize::letterbox(shape[1], shape[2], SIZE, SIZE, false)?;
        let canvas = self
            .images
            .resize_rgb_fragment(image.desc(), geometry)?
            .record(graph, std::slice::from_ref(image))?
            .remove(0);
        self.record_letterboxed(graph, &canvas, &self.metadata(&shapes)?, options)
    }

    /// Compose CNN and candidate decoding over caller-prepared canvases.
    /// Metadata is runtime F32 `[batch,3]`: scale, original width/2 (integer
    /// division), original height/2. The caller supplies original shapes at NMS.
    pub fn record_letterboxed(
        &self,
        graph: &mut Graph,
        canvases: &DeviceTensor,
        metadata: &DeviceTensor,
        options: DetectionOptions,
    ) -> Result<postprocess::RecordedDetections> {
        options.validate()?;
        let mut heads = self.record_heads(graph, canvases)?;
        heads.push(metadata.clone());
        let outputs = self
            .postprocess
            .fragment(canvases.desc().shape()[0], options)?
            .record(graph, &heads)?;
        Ok(postprocess::RecordedDetections { outputs, options })
    }
    /// Decode on GPU, select on CPU using compact scores/boxes, then gather
    /// resident detections. Waits for candidate metadata, not final row readback.
    pub fn decode_heads(
        &self,
        heads: &[DeviceTensor],
        scales: &[f32],
        shapes: &[[usize; 2]],
        options: DetectionOptions,
    ) -> Result<postprocess::Detections> {
        self.postprocess.submit(heads, scales, shapes, options)
    }
    /// Run inference, GPU decode, CPU NMS/ranking and asynchronous GPU gathering.
    pub fn submit(
        &self,
        canvases: &DeviceTensor,
        scales: &[f32],
        shapes: &[[usize; 2]],
        options: DetectionOptions,
    ) -> Result<postprocess::Detections> {
        options.validate()?;
        self.context().validate(canvases)?;
        let desc = canvases.desc();
        ensure!(
            desc.dtype() == DType::U8
                && desc.layout() == Layout::Nhwc
                && desc.is_contiguous()
                && desc.shape().len() == 4
                && desc.shape()[1..] == [SIZE, SIZE, 3],
            "expected contiguous NHWC uint8 640×640 canvases"
        );
        let batch = desc.shape()[0];
        ensure!(
            (1..=self.cnn.max_batch).contains(&batch)
                && scales.len() == batch
                && shapes.len() == batch
                && scales.iter().all(|s| s.is_finite() && *s > 0.)
                && shapes.iter().all(|s| s[0] > 0 && s[1] > 0),
            "invalid image scales or shapes"
        );
        let values: Vec<f32> = scales
            .iter()
            .zip(shapes)
            .flat_map(|(&scale, shape)| [scale, (shape[0] / 2) as f32, (shape[1] / 2) as f32])
            .collect();
        let metadata = self.context().upload(
            TensorDesc::new(DType::F32, vec![batch, 3])?,
            bytemuck::cast_slice(&values),
        )?;
        let decoded = self
            .prepare_letterboxed(batch, options)?
            .submit(&[canvases.clone(), metadata])?;
        self.postprocess.finish_resident(decoded, shapes, options)
    }
    /// Warm synchronized host latency, including preprocessing, transfers and decode.
    pub fn benchmark(
        &self,
        image: Image<'_>,
        samples: usize,
    ) -> Result<hrx::benchmark::Distribution> {
        ensure!(samples >= 10, "use at least ten timing samples");
        for _ in 0..10 {
            self.detect(image, DetectionOptions::default())?;
        }
        let mut times = Vec::with_capacity(samples);
        for _ in 0..samples {
            let start = std::time::Instant::now();
            self.detect(image, DetectionOptions::default())?;
            times.push(start.elapsed().as_secs_f64() * 1000.);
        }
        Ok(hrx::benchmark::Distribution::from_samples(times)?)
    }
    pub fn detect(&self, image: Image<'_>, options: DetectionOptions) -> Result<Vec<Detection>> {
        Ok(self.detect_batch(&[image], options)?.remove(0))
    }
    fn prepare_images(
        &self,
        shapes: &[[usize; 2]],
        options: DetectionOptions,
    ) -> Result<std::sync::Arc<PreparedModel>> {
        Ok(self.image_plans.get_or_prepare(
            (
                shapes.to_vec(),
                options.threshold.to_bits(),
                options.max_candidates.min(16800),
            ),
            || {
                let decode = self
                    .postprocess
                    .fragment(shapes.len(), options)
                    .map_err(|e| hrx::Error::Message(e.to_string()))?;
                let metadata = self.metadata(shapes)?;
                let cnn = self
                    .cnn
                    .fragment(shapes.len())
                    .map_err(|error| hrx::Error::Message(error.to_string()))?;
                let resizes = shapes
                    .chunk_by(|a, b| a == b)
                    .map(|group| {
                        let [width, height] = group[0];
                        let desc = TensorDesc::new(DType::U8, vec![group.len(), height, width, 3])?
                            .with_layout(Layout::Nhwc)?;
                        let resize = RgbResize::letterbox(height, width, SIZE, SIZE, false)?;
                        Ok((
                            desc.clone(),
                            self.images.resize_rgb_fragment(&desc, resize)?,
                        ))
                    })
                    .collect::<hrx::Result<Vec<_>>>()?;
                PreparedModel::prepare(self.context(), 3, |context| {
                    let mut graph = context.runtime().graph();
                    let mut inputs = Vec::with_capacity(shapes.len());
                    let mut resized = Vec::with_capacity(shapes.len());
                    for (desc, resize) in &resizes {
                        let input =
                            context.allocate_with(desc.clone(), MemoryPlacement::HostVisible)?;
                        resized.push(
                            resize
                                .record(&mut graph, std::slice::from_ref(&input))?
                                .remove(0),
                        );
                        // The API still accepts separate byte slices; input views
                        // pack equal-sized images directly into one batched allocation.
                        let shape = desc.shape();
                        let image_desc =
                            TensorDesc::new(DType::U8, vec![1, shape[1], shape[2], 3])?
                                .with_layout(Layout::Nhwc)?;
                        for i in 0..shape[0] {
                            inputs.push(input.view(i * image_desc.bytes(), image_desc.clone())?);
                        }
                    }
                    let canvases = if resized.len() == 1 {
                        resized.remove(0)
                    } else {
                        let canvases = context.allocate(
                            TensorDesc::new(DType::U8, vec![shapes.len(), SIZE, SIZE, 3])?
                                .with_layout(Layout::Nhwc)?,
                        )?;
                        let destination = canvases.binding().unwrap();
                        let mut start = 0;
                        for image in &resized {
                            let end = start + image.desc().bytes();
                            graph.copy(destination.slice(start..end)?, image.binding().unwrap())?;
                            start = end;
                        }
                        canvases
                    };
                    let mut heads = cnn.record(&mut graph, &[canvases])?;
                    heads.push(metadata.clone());
                    let outputs = decode.record(&mut graph, &heads)?;
                    Ok(InferenceGraph {
                        inputs,
                        outputs,
                        graph: graph.prepare()?,
                    })
                })
            },
        )?)
    }
    fn metadata(&self, shapes: &[[usize; 2]]) -> hrx::Result<DeviceTensor> {
        let mut values = Vec::with_capacity(shapes.len() * 3);
        for &[width, height] in shapes {
            let geometry = RgbResize::letterbox(height, width, SIZE, SIZE, false)?;
            values.extend([
                geometry.region[3] as f32 / height as f32,
                (width / 2) as f32,
                (height / 2) as f32,
            ]);
        }
        self.context().upload(
            TensorDesc::new(DType::F32, vec![shapes.len(), 3])?,
            bytemuck::cast_slice(&values),
        )
    }
    fn prepare_letterboxed(
        &self,
        batch: usize,
        options: DetectionOptions,
    ) -> Result<std::sync::Arc<PreparedModel>> {
        Ok(self.boxed_plans.get_or_prepare(
            (
                batch,
                options.threshold.to_bits(),
                options.max_candidates.min(16800),
            ),
            || {
                let cnn = self
                    .cnn
                    .fragment(batch)
                    .map_err(|e| hrx::Error::Message(e.to_string()))?;
                let decoder = self
                    .postprocess
                    .fragment(batch, options)
                    .map_err(|e| hrx::Error::Message(e.to_string()))?;
                PreparedModel::prepare(self.context(), 3, |context| {
                    let canvas = context.allocate_with(
                        TensorDesc::new(DType::U8, vec![batch, SIZE, SIZE, 3])?
                            .with_layout(Layout::Nhwc)?,
                        MemoryPlacement::HostVisible,
                    )?;
                    let metadata = context.allocate_with(
                        TensorDesc::new(DType::F32, vec![batch, 3])?,
                        MemoryPlacement::HostVisible,
                    )?;
                    let mut graph = context.runtime().graph();
                    let mut heads = cnn.record(&mut graph, std::slice::from_ref(&canvas))?;
                    heads.push(metadata.clone());
                    let outputs = decoder.record(&mut graph, &heads)?;
                    Ok(InferenceGraph {
                        inputs: vec![canvas, metadata],
                        outputs,
                        graph: graph.prepare()?,
                    })
                })
            },
        )?)
    }
    pub fn detect_batch(
        &self,
        images: &[Image<'_>],
        options: DetectionOptions,
    ) -> Result<Vec<Vec<Detection>>> {
        options.validate()?;
        for image in images {
            image.validate()?;
        }
        let mut out = Vec::with_capacity(images.len());
        for chunk in images.chunks(self.cnn.max_batch) {
            let shapes = chunk
                .iter()
                .map(|i| [i.width, i.height])
                .collect::<Vec<_>>();
            let inputs: Vec<_> = chunk.iter().map(|image| image.rgb).collect();
            let decoded = self
                .prepare_images(&shapes, options)?
                .acquire_blocking()?
                .submit_host(&inputs)?;
            out.extend(self.postprocess.finish_host(decoded, &shapes, options)?);
        }
        Ok(out)
    }
    /// Top-left-aligned 640×640 RGB canvases, their resize scales and original `[width, height]` values.
    pub fn detect_letterboxed(
        &self,
        canvases: &[u8],
        scales: &[f32],
        shapes: &[[usize; 2]],
        options: DetectionOptions,
    ) -> Result<Vec<Vec<Detection>>> {
        options.validate()?;
        ensure!(
            canvases.len().is_multiple_of(SIZE * SIZE * 3),
            "invalid canvases"
        );
        let b = canvases.len() / (SIZE * SIZE * 3);
        ensure!(
            scales.len() == b
                && shapes.len() == b
                && scales.iter().all(|s| s.is_finite() && *s > 0.)
                && shapes.iter().all(|s| s[0] > 0 && s[1] > 0),
            "invalid image scales or shapes"
        );
        let mut out = vec![];
        for (chunk_index, chunk) in canvases
            .chunks(self.cnn.max_batch * SIZE * SIZE * 3)
            .enumerate()
        {
            let batch = chunk.len() / (SIZE * SIZE * 3);
            let start = chunk_index * self.cnn.max_batch;
            let chunk_shapes = &shapes[start..start + batch];
            let metadata: Vec<f32> = scales[start..start + batch]
                .iter()
                .zip(chunk_shapes)
                .flat_map(|(&scale, shape)| [scale, (shape[0] / 2) as f32, (shape[1] / 2) as f32])
                .collect();
            let decoded = self
                .prepare_letterboxed(batch, options)?
                .acquire_blocking()?
                .submit_host(&[chunk, bytemuck::cast_slice(&metadata)])?;
            out.extend(
                self.postprocess
                    .finish_host(decoded, chunk_shapes, options)?,
            );
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod reference;
