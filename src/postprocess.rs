//! GPU candidate decoding, CPU selection, and resident landmark gathering.
use crate::{Detection, DetectionOptions, detection::select_candidates};
use anyhow::{Result, ensure};
use hrx::{
    inference::{ModelContext, PreparedModel},
    loom::Specialization,
    model::{Command, Dispatch, ModelSession},
    plan_cache::PlanCache,
    tensor::{DType, DeviceTensor, Layout, TensorDesc, TensorOps},
};
use std::sync::Arc;

type Key = (usize, u32, usize);
pub(crate) struct Postprocess {
    context: ModelContext,
    plans: PlanCache<Key, PreparedModel>,
    tensors: TensorOps,
}
impl Postprocess {
    pub fn new(context: &ModelContext) -> Result<Self> {
        Ok(Self {
            context: context.clone(),
            plans: PlanCache::new(8, PreparedModel::is_idle)?,
            tensors: TensorOps::new(context, 16)?,
        })
    }
    fn prepare(&self, batch: usize, options: DetectionOptions) -> Result<Arc<PreparedModel>> {
        options.validate()?;
        ensure!((1..=64).contains(&batch), "invalid postprocess batch");
        let capacity = options.max_candidates.min(16800);
        Ok(self
            .plans
            .get_or_prepare((batch, options.threshold.to_bits(), capacity), || {
                let mut session = ModelSession::in_context(&self.context)?;
                let mut inputs = Vec::new();
                for size in [80, 40, 20] {
                    let desc = TensorDesc::new(DType::F16, vec![batch, size, size, 64])?
                        .with_layout(Layout::Nhwc)?;
                    inputs.push((session.allocate(desc.bytes())?, desc));
                }
                let meta = TensorDesc::new(DType::F32, vec![batch, 3])?;
                inputs.push((session.allocate(meta.bytes())?, meta));
                let rows = TensorDesc::new(DType::F32, vec![batch * capacity, 16])?;
                let status = TensorDesc::new(DType::I32, vec![batch, 2])?;
                let compact = TensorDesc::new(DType::F32, vec![batch, capacity, 5])?;
                let candidate = session.allocate(rows.bytes())?;
                let flags = session.allocate(status.bytes())?;
                let summary = session.allocate(compact.bytes())?;
                let sigmoid: Vec<f32> = (0..=u16::MAX)
                    .map(|bits| 1. / (1. + (-half::f16::from_bits(bits).to_f32()).exp()))
                    .collect();
                let lut = session.weight(bytemuck::cast_slice(&sigmoid))?;
                let specialize = |source: &str| {
                    let mut source = source.to_owned();
                    for (key, value) in [
                        ("BATCH", batch.to_string()),
                        ("LAST_BATCH", (batch - 1).to_string()),
                        ("CAP", capacity.to_string()),
                        ("LAST_CAP", (capacity - 1).to_string()),
                        ("THRESHOLD", format!("{:.9e}", options.threshold)),
                        ("COUNT", compact.elements().to_string()),
                        ("LAST", (compact.elements() - 1).to_string()),
                        ("GRID", compact.elements().div_ceil(256).to_string()),
                    ] {
                        source = source.replace(&format!("@{key}@"), &value);
                    }
                    source
                };
                let decode = specialize(include_str!("../kernels/decode.loom"));
                let compact_code = specialize(include_str!("../kernels/compact.loom"));
                let kernels = unsafe {
                    session.compile(&[
                        (&decode, Specialization::new("decode")),
                        (&compact_code, Specialization::new("compact")),
                    ])?
                };
                let mut args: Vec<_> = inputs.iter().map(|(r, _)| r.read()).collect();
                args.extend([lut.read(), candidate.write(), flags.write()]);
                let commands = [
                    Command::Fill {
                        region: candidate,
                        value: 0,
                    },
                    Command::Dispatch(Dispatch::indices(
                        kernels[0],
                        [0],
                        [batch as u32, 1, 1],
                        args,
                    )),
                    Command::Dispatch(Dispatch::indices(
                        kernels[1],
                        [0],
                        [compact.elements().div_ceil(256) as u32, 1, 1],
                        vec![candidate.read(), summary.write()],
                    )),
                ];
                unsafe {
                    session.freeze(&self.context)?.prepare(
                        &commands,
                        &inputs,
                        &[(flags, status), (candidate, rows), (summary, compact)],
                        3,
                    )
                }
            })?)
    }
    /// Wait for compact candidate metadata, select on CPU, then enqueue GPU
    /// gathering. This intentional host stage is not a failure fallback.
    pub fn submit(
        &self,
        heads: &[DeviceTensor],
        scales: &[f32],
        shapes: &[[usize; 2]],
        options: DetectionOptions,
    ) -> Result<Detections> {
        ensure!(
            heads.len() == 3 && !scales.is_empty() && scales.len() == shapes.len(),
            "invalid detection metadata"
        );
        ensure!(
            scales.iter().all(|s| s.is_finite() && *s > 0.)
                && shapes.iter().all(|s| s[0] > 0 && s[1] > 0),
            "invalid image scales or shapes"
        );
        for head in heads {
            self.context.validate(head)?;
        }
        let plan = self.prepare(scales.len(), options)?;
        let capacity = options.max_candidates.min(16800);
        let metadata: Vec<f32> = scales
            .iter()
            .zip(shapes)
            .flat_map(|(scale, shape)| [*scale, (shape[0] / 2) as f32, (shape[1] / 2) as f32])
            .collect();
        let mut inputs = heads.to_vec();
        inputs.push(self.context.upload(
            TensorDesc::new(DType::F32, vec![scales.len(), 3])?,
            bytemuck::cast_slice(&metadata),
        )?);
        let decoded = plan.submit(&inputs)?;
        let status = self.context.download(&decoded.outputs()[0])?.wait()?;
        let mut candidates = Vec::with_capacity(scales.len());
        for pair in status.as_chunks::<8>().0 {
            let error = i32::from_le_bytes(pair[..4].try_into().unwrap());
            let count = i32::from_le_bytes(pair[4..].try_into().unwrap());
            ensure!(error & 1 == 0, "non-finite detector logit");
            ensure!(error & 2 == 0, "non-finite detector output");
            ensure!(error & 4 == 0, "candidate capacity exceeded");
            ensure!(
                error == 0 && count >= 0 && count as usize <= capacity,
                "invalid detector status"
            );
            candidates.push(count as usize);
        }
        let reads = candidates
            .iter()
            .enumerate()
            .map(|(b, &count)| {
                let view = decoded.outputs()[2].view(
                    b * capacity * 20,
                    TensorDesc::new(DType::F32, vec![count, 5])?,
                )?;
                self.context.download(&view)
            })
            .collect::<hrx::Result<Vec<_>>>()?;
        let mut selected = Vec::new();
        let mut offsets = vec![0];
        for (b, read) in reads.into_iter().enumerate() {
            let bytes = read.wait()?;
            let boxes: Vec<[f32; 5]> = bytes
                .as_chunks::<20>()
                .0
                .iter()
                .map(|row| {
                    std::array::from_fn(|i| {
                        f32::from_le_bytes(row[i * 4..][..4].try_into().unwrap())
                    })
                })
                .collect();
            selected.extend(
                select_candidates(&boxes, shapes[b], options)
                    .into_iter()
                    .map(|i| b * capacity + i),
            );
            offsets.push(selected.len());
        }
        let rows = self.tensors.gather_rows(&decoded.outputs()[1], &selected)?;
        Ok(Detections {
            context: self.context.clone(),
            rows,
            offsets,
            candidates,
        })
    }
}

/// Selected resident F32 [total_faces,16] rows: score, bbox[4], landmarks[10],
/// zero padding. CPU selection downloads only candidate scores/boxes and status.
/// Image row offsets preserve batch order; gathering is asynchronous.
pub struct Detections {
    context: ModelContext,
    rows: DeviceTensor,
    offsets: Vec<usize>,
    candidates: Vec<usize>,
}
impl Detections {
    pub fn rows(&self) -> &DeviceTensor {
        &self.rows
    }
    pub fn completion(&self) -> &hrx::Completion {
        self.rows.completion()
    }
    /// Candidate counts before suppression, useful for transfer accounting.
    pub fn candidate_counts(&self) -> &[usize] {
        &self.candidates
    }
    /// Wait for gathering and return the number of selected rows per image.
    pub fn counts(&self) -> Result<Vec<usize>> {
        self.completion().wait()?;
        Ok(self.offsets.windows(2).map(|w| w[1] - w[0]).collect())
    }
    /// Checked strided [faces,5,2] view without downloading coordinates.
    pub fn landmarks(&self, image: usize, range: std::ops::Range<usize>) -> Result<DeviceTensor> {
        ensure!(
            image < self.offsets.len() - 1
                && range.start < range.end
                && range.end <= self.offsets[image + 1] - self.offsets[image],
            "invalid detection landmark range"
        );
        Ok(self.rows.view(
            (self.offsets[image] + range.start) * 64 + 20,
            TensorDesc::strided(DType::F32, vec![range.len(), 5, 2], vec![16, 2, 1])?,
        )?)
    }
    /// Download only selected full rows, never the dense network heads.
    pub fn wait(self) -> Result<Vec<Vec<Detection>>> {
        let bytes = self.context.download(&self.rows)?.wait()?;
        Ok(self
            .offsets
            .windows(2)
            .map(|w| {
                bytes[w[0] * 64..w[1] * 64]
                    .as_chunks::<64>()
                    .0
                    .iter()
                    .map(|row| {
                        let v: [f32; 15] = std::array::from_fn(|i| {
                            f32::from_le_bytes(row[i * 4..][..4].try_into().unwrap())
                        });
                        Detection {
                            score: v[0],
                            bbox: v[1..5].try_into().unwrap(),
                            landmarks: std::array::from_fn(|i| [v[5 + 2 * i], v[6 + 2 * i]]),
                        }
                    })
                    .collect()
            })
            .collect())
    }
}
