use anyhow::{Context, Result, ensure};
use hrx::model::ScratchPlanner;
use std::collections::{HashMap, HashSet};
#[derive(Clone, Debug, Default)]
pub(crate) struct Op {
    pub kind: &'static str,
    pub variant: &'static str,
    pub name: String,
    pub src: String,
    pub dst: String,
    pub extra: String,
    pub h: usize,
    pub w: usize,
    pub stride: usize,
    pub cin_pad: usize,
    pub cin_stride: usize,
    pub k: usize,
    /// Output channels packed in the weight matrix, rounded for WMMA tiles.
    pub n: usize,
    /// Physical output activation channels; independent of the weight tile width.
    pub cout_stride: usize,
    pub ho: usize,
    pub wo: usize,
    pub tile: usize,
    pub splits: usize,
    pub slope: bool,
    pub bytes: usize,
    pub src_buf: usize,
    pub dst_buf: usize,
    pub extra_buf: usize,
}
pub(crate) struct Plan {
    pub ops: Vec<Op>,
    pub buffers: Vec<usize>,
    pub weights: HashMap<String, Vec<u8>>,
    pub outputs: Vec<usize>,
}
pub(crate) fn align(n: usize, a: usize) -> usize {
    n.div_ceil(a) * a
}
pub(crate) fn storage(c: usize) -> usize {
    if c <= 8 { 8 } else { align(c, 32) }
}
pub(crate) fn emit32(weights: &mut HashMap<String, Vec<u8>>, name: String, v: &[f64]) {
    weights.insert(
        name,
        v.iter().flat_map(|x| (*x as f32).to_le_bytes()).collect(),
    );
}
pub(crate) fn emit16(weights: &mut HashMap<String, Vec<u8>>, name: String, v: &[f64]) {
    weights.insert(
        name,
        v.iter()
            .flat_map(|x| half::f16::from_f64(*x).to_le_bytes())
            .collect(),
    );
}
pub(crate) fn pack(
    w: &[f64],
    co: usize,
    ci: usize,
    taps: usize,
    cp: usize,
    n: usize,
    k: usize,
) -> Vec<f64> {
    let mut out = vec![0.; n * k];
    for o in 0..co {
        for c in 0..ci {
            for t in 0..taps {
                out[o * k + t * cp + c] = w[(o * ci + c) * taps + t];
            }
        }
    }
    out
}
pub(crate) fn finish(
    mut pending: Vec<Op>,
    aliases: HashMap<String, String>,
    weights: HashMap<String, Vec<u8>>,
    outputs: &[String],
) -> Result<Plan> {
    let resolve = |name: &str| -> Result<String> {
        let mut s = name.to_owned();
        let mut seen = HashSet::new();
        while let Some(v) = aliases.get(&s) {
            ensure!(seen.insert(s.clone()), "cyclic tensor alias");
            s = v.clone();
        }
        Ok(s)
    };
    for l in &mut pending {
        if l.kind != "convert" {
            l.src = resolve(&l.src)?;
        }
        if !l.extra.is_empty() {
            l.extra = resolve(&l.extra)?;
        }
    }
    let mut ready = HashSet::from([pending[0].src.clone()]);
    let mut ops = vec![];
    while !pending.is_empty() {
        let i = pending
            .iter()
            .position(|l| {
                ready.contains(&l.src) && (l.extra.is_empty() || ready.contains(&l.extra))
            })
            .context("cyclic or unresolved model schedule")?;
        let op = pending.remove(i);
        ensure!(ready.insert(op.dst.clone()), "duplicate launch output");
        ops.push(op);
    }
    let mut last = HashMap::new();
    for (i, l) in ops.iter().enumerate() {
        last.insert(l.src.clone(), i);
        if !l.extra.is_empty() {
            last.insert(l.extra.clone(), i);
        }
    }
    for o in outputs {
        last.insert(o.clone(), ops.len());
    }
    let mut planner = ScratchPlanner::new();
    for (i, op) in ops.iter().enumerate() {
        planner.insert(
            op.dst.clone(),
            op.bytes,
            i,
            last.get(&op.dst).copied().context("unused model value")?,
        )?;
    }
    let allocation = planner.finish();
    for l in &mut ops {
        l.src_buf = allocation
            .assignments()
            .get(&l.src)
            .copied()
            .unwrap_or(usize::MAX);
        l.extra_buf = allocation
            .assignments()
            .get(&l.extra)
            .copied()
            .unwrap_or(usize::MAX);
        l.dst_buf = allocation.slot(&l.dst)?;
        ensure!(
            l.dst_buf != l.src_buf && l.dst_buf != l.extra_buf,
            "in-place activation hazard"
        );
    }
    let outputs = outputs
        .iter()
        .map(|o| allocation.slot(o).map_err(anyhow::Error::from))
        .collect::<Result<_>>()?;
    Ok(Plan {
        ops,
        buffers: allocation.slots().to_vec(),
        weights,
        outputs,
    })
}
