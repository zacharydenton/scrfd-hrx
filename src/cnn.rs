use crate::plan::{Op, Plan};
use anyhow::{Result, ensure};
use hrx::{
    inference::{ModelContext, PreparedModel},
    loom::Specialization,
    model::{Command, Dispatch, KernelId, ModelDefinition, ModelSession, Region},
    plan_cache::PlanCache,
    tensor::{DType, Layout, TensorDesc},
};
use std::collections::HashMap;
pub(crate) struct Cnn {
    pub engine: ModelDefinition,
    plans: PlanCache<usize, PreparedModel>,
    pub input: Region,
    pub outputs: Vec<Region>,
    pub max_batch: usize,
    ops: Vec<Op>,
    buffers: Vec<Region>,
    weights: HashMap<String, Region>,
    kernels: Vec<KernelId>,
}
impl Cnn {
    pub fn new(mut plan: Plan, context: &ModelContext, max_batch: usize) -> Result<Self> {
        ensure!((1..=64).contains(&max_batch), "max_batch must be 1..=64");
        let device = context.runtime().gpu()?;
        ensure!(
            device.target().as_str() == "gfx1151",
            "SCRFD requires gfx1151"
        );
        let mut engine = ModelSession::in_context(context)?;
        let mut weights = HashMap::new();
        for (name, data) in plan.weights {
            weights.insert(name, engine.weight(&data)?);
        }
        let input = engine.allocate_shared(max_batch * SIZE * SIZE * 3)?;
        // Keep terminal outputs separate from the device-local activation pool.
        // The original liveness plan may reuse these slots for earlier layers.
        for output in &mut plan.outputs {
            let last = plan
                .ops
                .iter_mut()
                .rfind(|op| op.dst_buf == *output)
                .unwrap();
            *output = plan.buffers.len();
            plan.buffers.push(last.bytes);
            last.dst_buf = *output;
        }
        let buffers = plan
            .buffers
            .iter()
            .enumerate()
            .map(|(i, n)| {
                if plan.outputs.contains(&i) {
                    engine.allocate_shared(n * max_batch)
                } else {
                    engine.allocate(n * max_batch)
                }
            })
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let outputs = plan.outputs.iter().map(|i| buffers[*i]).collect();
        let specs = plan.ops.iter().map(specification).collect::<Vec<_>>();
        // Every source is embedded in this crate and its bindings are declared below.
        let kernels = unsafe { engine.compile(&specs)? };
        let engine = engine.freeze(context)?;
        Ok(Self {
            engine,
            plans: PlanCache::new(8, PreparedModel::is_idle)?,
            input,
            outputs,
            max_batch,
            ops: plan.ops,
            buffers,
            weights,
            kernels,
        })
    }
    pub fn prepare(&self, batch: usize) -> Result<std::sync::Arc<PreparedModel>> {
        ensure!((1..=self.max_batch).contains(&batch), "invalid batch size");
        Ok(self.plans.get_or_prepare(batch, || {
            let shaped = |region: Region| region.slice(0, region.len() / self.max_batch * batch);
            let mut commands = vec![];
            for (kernel, l) in self.ops.iter().enumerate() {
                let src = if l.src_buf == usize::MAX {
                    shaped(self.input)?
                } else {
                    shaped(self.buffers[l.src_buf])?
                };
                let dst = shaped(self.buffers[l.dst_buf])?;
                let m = batch * l.ho * l.wo;
                let (scalar, grid, bindings) = match l.kind {
                    "convert" => (
                        batch * l.h,
                        [batch as u32 * l.h as u32, 1, 1],
                        vec![src, dst],
                    ),
                    "pool" => (m, [(batch * l.ho) as u32, 1, 1], vec![src, dst]),
                    "reduce" => (
                        batch,
                        [batch as u32, 1, 1],
                        vec![src, self.weights[&(l.name.clone() + "_b")], dst],
                    ),
                    _ => {
                        let mut args = vec![
                            src,
                            self.weights[&l.name],
                            self.weights[&(l.name.clone() + "_b")],
                            dst,
                        ];
                        if l.extra_buf != usize::MAX {
                            args.push(shaped(self.buffers[l.extra_buf])?);
                        }
                        if l.slope {
                            args.push(self.weights[&(l.name.clone() + "_slope")]);
                        }
                        let m = if l.kind == "head" { batch } else { m };
                        (
                            m,
                            [
                                (if l.tile == 32 {
                                    // The narrow kernel computes 32 channels and
                                    // zeroes the rest of its 64-channel storage.
                                    1
                                } else {
                                    l.n / if l.kind == "conv" { l.tile } else { 64 }
                                }) as u32,
                                m.div_ceil(64) as u32,
                                if l.kind == "head" { l.splits as u32 } else { 1 },
                            ],
                            args,
                        )
                    }
                };
                let bindings = bindings
                    .into_iter()
                    .map(|region| {
                        if region == dst {
                            region.write()
                        } else {
                            region.read()
                        }
                    })
                    .collect();
                commands.push(Command::Dispatch(Dispatch::indices(
                    self.kernels[kernel],
                    [scalar as u32],
                    grid,
                    bindings,
                )));
            }
            // Plans reject in-place activation hazards; each output above is the
            // complete write set of its embedded kernel.
            let outputs = self
                .outputs
                .iter()
                .zip([80, 40, 20])
                .map(|(&region, size)| {
                    Ok((
                        region,
                        TensorDesc::new(DType::F16, vec![batch, size, size, 64])?
                            .with_layout(Layout::Nhwc)?,
                    ))
                })
                .collect::<hrx::Result<Vec<_>>>()?;
            unsafe {
                self.engine.prepare(
                    &commands,
                    &[(
                        self.input,
                        TensorDesc::new(DType::U8, vec![batch, SIZE, SIZE, 3])?
                            .with_layout(Layout::Nhwc)?,
                    )],
                    &outputs,
                    3,
                )
            }
        })?)
    }
}
fn specification(l: &Op) -> (&'static str, Specialization) {
    let (source, symbol, namespace) = match l.kind {
        "convert" => (
            "hwc_u8_to_nhwc_f16".to_owned(),
            format!("{MODEL}_hwc_u8_to_nhwc_f16"),
            format!("{MODEL}.hwc_u8_to_nhwc_f16"),
        ),
        "pool" => (
            "pool2_f16".into(),
            "scrfd_pool2_f16".into(),
            "scrfd.pool2_f16".into(),
        ),
        "head" => (
            "matmul_splitk_f16_wmma".into(),
            "arcface_matmul_splitk_f16_wmma".into(),
            "arcface.matmul_splitk_f16_wmma".into(),
        ),
        "reduce" => (
            "splitk_reduce_f32".into(),
            "arcface_splitk_reduce_f32".into(),
            "arcface.splitk_reduce_f32".into(),
        ),
        "matmul" if l.variant == "plain" => (
            "matmul_bias_f16_wmma_af16_cf16".into(),
            "dinov3_matmul_bias_f16_wmma_af16_cf16".into(),
            "dinov3.matmul_bias_f16_wmma_af16_cf16".into(),
        ),
        "matmul" => (
            "matmul_add_resized_f16_wmma".into(),
            "scrfd_matmul_add_resized_f16_wmma".into(),
            "scrfd.matmul_add_resized_f16_wmma".into(),
        ),
        _ => {
            let base = if l.tile == 32 {
                "conv3x3_n32_f16_wmma"
            } else if l.tile == 128 {
                "conv3x3_n128_f16_wmma"
            } else {
                "conv3x3_f16_wmma"
            };
            let src = if l.variant == "plain" {
                base.to_owned()
            } else {
                format!("{base}_{}", l.variant)
            };
            let symbol = format!("{MODEL}_{src}");
            let ns = format!("{MODEL}.{src}");
            (src, symbol, ns)
        }
    };
    let mut spec = Specialization::new(symbol);
    let mut put = |name: &str, value: usize| {
        spec.set_config(format!("{namespace}.{name}"), value.to_string());
    };
    match l.kind {
        "convert" => put("size", SIZE),
        "pool" => {
            put("height", l.h);
            put("width", l.w);
            put("channels", l.cin_stride);
            put("take_max", usize::from(l.variant == "max"));
        }
        "head" => {
            put("k_size", l.k);
            put("n_size", l.n);
            put("splits", l.splits);
        }
        "reduce" => {
            put("n_size", l.n);
            put("splits", l.splits);
        }
        _ => {
            put("k_size", l.k);
            put("n_size", l.n);
            if l.kind == "conv" || l.variant == "add_resized" {
                put("height", l.h);
                put("width", l.w);
            }
            if l.kind == "conv" {
                put("stride", l.stride);
                put("cin_pad", l.cin_pad);
                put("cin_stride", l.cin_stride);
            }
        }
    }
    (kernel_source(&source), spec)
}

const SIZE: usize = 640;
const MODEL: &str = "scrfd";
fn kernel_source(name: &str) -> &'static str {
    match name {
        "conv3x3_n32_f16_wmma_relu" => include_str!("../kernels/conv3x3_n32_f16_wmma_relu.loom"),
        "conv3x3_f16_wmma" => include_str!("../kernels/conv3x3_f16_wmma.loom"),
        "conv3x3_f16_wmma_add" => include_str!("../kernels/conv3x3_f16_wmma_add.loom"),
        "conv3x3_f16_wmma_relu" => include_str!("../kernels/conv3x3_f16_wmma_relu.loom"),
        "conv3x3_f16_wmma_relu_add" => include_str!("../kernels/conv3x3_f16_wmma_relu_add.loom"),
        "conv3x3_n128_f16_wmma_relu" => include_str!("../kernels/conv3x3_n128_f16_wmma_relu.loom"),
        "conv3x3_n128_f16_wmma_relu_add" => {
            include_str!("../kernels/conv3x3_n128_f16_wmma_relu_add.loom")
        }
        "hwc_u8_to_nhwc_f16" => include_str!("../kernels/hwc_u8_to_nhwc_f16.loom"),
        "matmul_add_resized_f16_wmma" => {
            include_str!("../kernels/matmul_add_resized_f16_wmma.loom")
        }
        "matmul_bias_f16_wmma_af16_cf16" => {
            include_str!("../kernels/matmul_bias_f16_wmma_af16_cf16.loom")
        }
        "pool2_f16" => include_str!("../kernels/pool2_f16.loom"),
        _ => unreachable!("unknown production kernel"),
    }
}
