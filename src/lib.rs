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
    image::{ImageOps, RgbResize},
    inference::{Inference, ModelContext},
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
        let heads = self.submit_heads(canvases)?;
        self.decode_heads(heads.outputs(), scales, shapes, options)
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
            let mut canvases = self.context().allocate(
                TensorDesc::new(DType::U8, vec![chunk.len(), SIZE, SIZE, 3])?
                    .with_layout(Layout::Nhwc)?,
            )?;
            let mut scales = Vec::with_capacity(chunk.len());
            for (index, image) in chunk.iter().enumerate() {
                let geometry = RgbResize::letterbox(image.height, image.width, SIZE, SIZE, false)?;
                let input = TensorDesc::new(DType::U8, vec![1, image.height, image.width, 3])?
                    .with_layout(Layout::Nhwc)?;
                let resized = self
                    .images
                    .prepare_resize_rgb(&input, geometry)?
                    .acquire_blocking()?
                    .submit_host(&[image.rgb])?;
                self.context().copy_into(
                    &mut canvases,
                    index * SIZE * SIZE * 3,
                    &resized.outputs()[0],
                )?;
                scales.push(geometry.region[3] as f32 / image.height as f32);
            }
            let shapes = chunk
                .iter()
                .map(|i| [i.width, i.height])
                .collect::<Vec<_>>();
            out.extend(self.submit(&canvases, &scales, &shapes, options)?.wait()?);
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
            let heads = self
                .cnn
                .prepare(batch)?
                .acquire_blocking()?
                .submit_host(&[chunk])?;
            let start = chunk_index * self.cnn.max_batch;
            out.extend(
                self.decode_heads(
                    heads.outputs(),
                    &scales[start..start + batch],
                    &shapes[start..start + batch],
                    options,
                )?
                .wait()?,
            );
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod reference;
