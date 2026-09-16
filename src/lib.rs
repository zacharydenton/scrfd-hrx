//! SCRFD det_10g face detection on packed RGB images. Boxes and landmarks
//! are returned in original-image coordinates.
mod cnn;
pub mod detection;
pub mod hub;
mod model;
mod onnx;
mod plan;
use anyhow::{Result, ensure};
pub use detection::{Detection, DetectionOptions, Image, Ranking};
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
/// Resident detector and stream. Calls require exclusive access.
pub struct Scrfd {
    cnn: cnn::Cnn,
    heads: Vec<Vec<half::f16>>,
    canvas: Vec<u8>,
    resizer: fast_image_resize::Resizer,
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
        ensure!(
            (1..=64).contains(&options.max_batch),
            "max_batch must be 1..=64"
        );
        Ok(Self {
            canvas: vec![0; options.max_batch * SIZE * SIZE * 3],
            resizer: fast_image_resize::Resizer::new(),
            cnn: cnn::Cnn::new(
                model::load(path.as_ref())?,
                options.device,
                options.max_batch,
            )?,
            heads: [80, 40, 20]
                .map(|s| vec![half::f16::ZERO; options.max_batch * s * s * 64])
                .into(),
        })
    }
    /// Compare resident graph replay and direct dispatch, excluding preprocessing and transfers.
    pub fn benchmark(&mut self, image: Image<'_>, samples: usize) -> Result<ForwardTimings> {
        self.detect(image, DetectionOptions::default())?;
        Ok(self.cnn.engine.benchmark(1, samples)?)
    }
    pub fn detect(
        &mut self,
        image: Image<'_>,
        options: DetectionOptions,
    ) -> Result<Vec<Detection>> {
        Ok(self.detect_batch(&[image], options)?.remove(0))
    }
    pub fn detect_batch(
        &mut self,
        images: &[Image<'_>],
        options: DetectionOptions,
    ) -> Result<Vec<Vec<Detection>>> {
        options.validate()?;
        for image in images {
            image.validate()?;
        }
        let mut canvas = std::mem::take(&mut self.canvas);
        let result = (|| -> Result<Vec<Vec<Detection>>> {
            let mut out = Vec::with_capacity(images.len());
            for chunk in images.chunks(self.cnn.max_batch) {
                let mut scales = Vec::with_capacity(chunk.len());
                for (image, dst) in chunk.iter().zip(canvas.chunks_exact_mut(SIZE * SIZE * 3)) {
                    scales.push(detection::letterbox_into(*image, dst, &mut self.resizer)?);
                }
                let shapes = chunk
                    .iter()
                    .map(|i| [i.width, i.height])
                    .collect::<Vec<_>>();
                out.extend(self.detect_letterboxed(
                    &canvas[..chunk.len() * SIZE * SIZE * 3],
                    &scales,
                    &shapes,
                    options,
                )?);
            }
            Ok(out)
        })();
        self.canvas = canvas;
        result
    }
    /// Top-left-aligned 640×640 RGB canvases, their resize scales and original `[width, height]` values.
    pub fn detect_letterboxed(
        &mut self,
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
            let batch = self.cnn.run(chunk)?;
            let mut outputs = self
                .heads
                .iter_mut()
                .zip([80, 40, 20])
                .zip(&self.cnn.outputs)
                .map(|((head, size), r)| {
                    (
                        *r,
                        bytemuck::cast_slice_mut(&mut head[..batch * size * size * 64]),
                    )
                })
                .collect::<Vec<_>>();
            self.cnn.engine.read_many(&mut outputs)?;
            for i in 0..batch {
                let j = chunk_index * self.cnn.max_batch + i;
                out.push(detection::decode(
                    &self.heads,
                    i,
                    scales[j],
                    shapes[j],
                    options,
                )?);
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests;

pub use hrx::loom::model::{Distribution, ForwardTimings};

#[cfg(test)]
mod reference;
