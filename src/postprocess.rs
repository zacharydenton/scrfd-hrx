//! GPU candidate decoding, CPU selection, and resident landmark gathering.
use crate::{Detection, DetectionOptions, detection::select_candidates};
use anyhow::{Result, ensure};
use hrx::{
    inference::{Inference, ModelContext, PreparedModel},
    loom::Specialization,
    model::{Command, Dispatch, ModelFragment, ModelSession},
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
        let capacity = options.max_candidates.min(16800);
        Ok(self
            .plans
            .get_or_prepare((batch, options.threshold.to_bits(), capacity), || {
                self.fragment(batch, options)
                    .map_err(|e| hrx::Error::Message(e.to_string()))?
                    .prepare(3)
            })?)
    }
    /// Shared decoding fragment for both host and resident entry points.
    pub(crate) fn fragment(
        &self,
        batch: usize,
        options: DetectionOptions,
    ) -> Result<ModelFragment> {
        options.validate()?;
        ensure!((1..=64).contains(&batch), "invalid postprocess batch");
        let capacity = options.max_candidates.min(16800);
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
        let candidate = session.allocate_shared(rows.bytes())?;
        let flags = session.allocate_shared(status.bytes())?;
        let summary = session.allocate_shared(compact.bytes())?;
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
            ] {
                source = source.replace(&format!("@{key}@"), &value);
            }
            source
        };
        // 66 blocks of 256 anchors: local ranks, block counts/error bits,
        // and exclusive block offsets preserve the CPU anchor order.
        let ranks = session.allocate(batch * 16896 * 4)?;
        let groups = session.allocate(batch * 66 * 2 * 4)?;
        let offsets = session.allocate(batch * 66 * 4)?;
        let classify = specialize(include_str!("../kernels/classify.loom"));
        let prefix = specialize(include_str!("../kernels/prefix_groups.loom"));
        let decode = specialize(include_str!("../kernels/decode_parallel.loom"));
        let kernels = unsafe {
            session.compile(&[
                (&classify, Specialization::new("classify")),
                (&prefix, Specialization::new("prefix_groups")),
                (&decode, Specialization::new("decode_parallel")),
            ])?
        };
        let commands = [
            Command::Dispatch(Dispatch::indices(
                kernels[0],
                [0],
                [66, batch as u32, 1],
                vec![
                    inputs[0].0.read(),
                    inputs[1].0.read(),
                    inputs[2].0.read(),
                    lut.read(),
                    ranks.write(),
                    groups.write(),
                ],
            )),
            Command::Dispatch(Dispatch::indices(
                kernels[1],
                [0],
                [1, batch as u32, 1],
                vec![groups.read(), offsets.write(), flags.write()],
            )),
            Command::Dispatch(Dispatch::indices(
                kernels[2],
                [0],
                [66, batch as u32, 1],
                vec![
                    inputs[0].0.read(),
                    inputs[1].0.read(),
                    inputs[2].0.read(),
                    inputs[3].0.read(),
                    lut.read(),
                    ranks.read(),
                    offsets.read(),
                    candidate.write(),
                    summary.write(),
                    flags.read_write(),
                ],
            )),
        ];
        unsafe {
            Ok(session.freeze(&self.context)?.fragment(
                &commands,
                &inputs,
                &[(flags, status), (candidate, rows), (summary, compact)],
            )?)
        }
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
        self.finish_resident(decoded, shapes, options)
    }
    pub(crate) fn finish_resident(
        &self,
        decoded: Inference,
        shapes: &[[usize; 2]],
        options: DetectionOptions,
    ) -> Result<Detections> {
        let selected = self.select(decoded, shapes, options)?;
        let rows = self
            .tensors
            .gather_rows(&selected.rows, &selected.indices)?;
        Ok(Detections {
            context: self.context.clone(),
            rows,
            offsets: selected.offsets,
            candidates: selected.candidates,
        })
    }
    fn select(
        &self,
        decoded: Inference,
        shapes: &[[usize; 2]],
        options: DetectionOptions,
    ) -> Result<Selected> {
        let outputs = decoded.wait()?;
        Self::select_outputs(&outputs, shapes, options)
    }
    fn select_outputs(
        outputs: &[DeviceTensor],
        shapes: &[[usize; 2]],
        options: DetectionOptions,
    ) -> Result<Selected> {
        options.validate()?;
        let capacity = options.max_candidates.min(16800);
        // Keep the inference lease until selection and any host row reads finish.
        // These terminal buffers are coherent; no dense readback/copy is needed.
        let status_binding = outputs[0].binding().unwrap();
        let status = status_binding.map_read()?;
        ensure!(
            status.len() == shapes.len() * 8,
            "invalid detector status shape"
        );
        let mut candidates = Vec::with_capacity(shapes.len());
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
        let summary_binding = outputs[2].binding().unwrap();
        let summary = summary_binding.map_read()?;
        let mut indices = Vec::new();
        let mut offsets = vec![0];
        for (b, &count) in candidates.iter().enumerate() {
            let boxes: &[[f32; 5]] =
                bytemuck::try_cast_slice(&summary[b * capacity * 20..][..count * 20])
                    .map_err(|e| anyhow::anyhow!("invalid candidate summary: {e}"))?;
            indices.extend(
                select_candidates(boxes, shapes[b], options)
                    .into_iter()
                    .map(|i| b * capacity + i),
            );
            offsets.push(indices.len());
        }
        Ok(Selected {
            rows: outputs[1].clone(),
            indices,
            offsets,
            candidates,
        })
    }
    pub(crate) fn finish_host(
        &self,
        decoded: Inference,
        shapes: &[[usize; 2]],
        options: DetectionOptions,
    ) -> Result<Vec<Vec<Detection>>> {
        let selected = self.select(decoded, shapes, options)?;
        let binding = selected.rows.binding().unwrap();
        let rows = binding.map_read()?;
        Ok(selected
            .offsets
            .windows(2)
            .map(|range| {
                selected.indices[range[0]..range[1]]
                    .iter()
                    .map(|&i| {
                        let row = &rows[i * 64..][..64];
                        let v: [f32; 15] = std::array::from_fn(|c| {
                            f32::from_le_bytes(row[c * 4..][..4].try_into().unwrap())
                        });
                        Detection {
                            score: v[0],
                            bbox: v[1..5].try_into().unwrap(),
                            landmarks: std::array::from_fn(|c| [v[5 + 2 * c], v[6 + 2 * c]]),
                        }
                    })
                    .collect()
            })
            .collect())
    }
}

/// Decoder bindings recorded into a caller-owned graph. No execution or copy
/// is performed by construction; keep this object with that prepared graph.
pub struct RecordedDetections {
    pub(crate) outputs: Vec<DeviceTensor>,
    pub(crate) options: DetectionOptions,
}
impl RecordedDetections {
    /// Dense candidate rows for direct GPU gathering in a downstream graph.
    pub fn rows(&self) -> &DeviceTensor {
        &self.outputs[1]
    }

    /// The explicit CPU control boundary. Wait for the graph execution that
    /// wrote these bindings, map only status and score/box summaries, and
    /// return row indices. Landmarks and image pixels are not read by the CPU.
    pub fn select(
        &self,
        completion: &hrx::Completion,
        shapes: &[[usize; 2]],
    ) -> Result<Vec<Vec<usize>>> {
        ensure!(
            shapes.iter().all(|shape| !shape.contains(&0)),
            "invalid image shapes"
        );
        completion.wait()?;
        let selected = Postprocess::select_outputs(&self.outputs, shapes, self.options)?;
        Ok(selected
            .offsets
            .windows(2)
            .map(|range| selected.indices[range[0]..range[1]].to_vec())
            .collect())
    }
}

struct Selected {
    rows: DeviceTensor,
    indices: Vec<usize>,
    offsets: Vec<usize>,
    candidates: Vec<usize>,
}

/// Selected resident F32 [total_faces,16] rows: score, bbox[4], landmarks[10],
/// zero padding. CPU selection maps coherent candidate scores/boxes and status.
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
