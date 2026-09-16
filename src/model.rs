use crate::{
    onnx::{Network, Node},
    plan::*,
};
use anyhow::{Context, Result, ensure};
use std::{
    collections::{HashMap, HashSet},
    path::Path,
};
pub(crate) fn load(path: &Path) -> Result<Plan> {
    let net = Network::load(path, 640)?;
    ensure!(
        net.outputs().len() == 9
            && net.nodes().iter().filter(|n| n.op_type() == "Conv").count() == 58,
        "expected det_10g graph"
    );
    let mut heads: HashMap<String, HashMap<usize, usize>> = HashMap::new();
    for (i, n) in net.nodes().iter().enumerate() {
        if n.op_type() == "Conv" {
            let co = net.tensor(&n.inputs()[1])?.shape()?[0];
            if [2, 8, 20].contains(&co) {
                ensure!(
                    heads
                        .entry(n.inputs()[0].clone())
                        .or_default()
                        .insert(co, i)
                        .is_none(),
                    "duplicate head"
                );
            }
        }
    }
    ensure!(
        heads.len() == 3 && heads.values().all(|g| g.len() == 3),
        "expected three score/box/landmark heads"
    );
    let members: HashSet<_> = heads.values().flat_map(|g| g.values().copied()).collect();
    // Only discard head operators whose semantics are implemented by packing
    // or decoding. Every terminal must be one of the nine declared outputs.
    let mut fused = HashSet::new();
    let mut head_scales = HashMap::new();
    let mut head_outputs = HashSet::new();
    for group in heads.values() {
        for (&channels, &i) in group {
            let (output, scale) = validate_head(&net, &net.nodes()[i], channels, &mut fused)?;
            ensure!(head_outputs.insert(output), "duplicate head output");
            head_scales.insert(i, scale);
        }
    }
    let declared_outputs: HashSet<_> = net.outputs().iter().cloned().collect();
    ensure!(
        head_outputs == declared_outputs,
        "expected only SCRFD head outputs"
    );
    let mut weights = HashMap::new();
    let mut aliases = HashMap::from([(net.inputs()[0].clone(), "nhwc_input".into())]);
    let mut ops = vec![Op {
        kind: "convert",
        name: "convert".into(),
        src: net.inputs()[0].clone(),
        dst: "nhwc_input".into(),
        h: 640,
        w: 640,
        ho: 640,
        wo: 640,
        bytes: 640 * 640 * 8 * 2,
        ..Default::default()
    }];
    let mut index = 0;
    for (i, node) in net.nodes().iter().enumerate() {
        match node.op_type() {
            "Conv" => {
                let name = format!("c{index:02}");
                index += 1;
                if members.contains(&i) {
                    continue;
                }
                let sh = net.tensor(&node.inputs()[1])?.shape()?;
                let (co, ci, taps) = (sh[0], sh[1], sh[2] * sh[3]);
                let mut variant = "plain";
                let mut extra = String::new();
                let mut output = node.output()?;
                let node_output = node.output()?;
                if let Some(add) = net
                    .consumers(node_output)
                    .find(|n| n.op_type() == "Add" && n.inputs()[0] == node_output)
                {
                    ensure!(
                        net.consumers(node.output()?).count() == 1,
                        "unfused conv consumer"
                    );
                    let other = net
                        .producer(&add.inputs()[1])
                        .context("missing residual producer")?;
                    if other.op_type() == "Resize" {
                        ensure!(
                            net.consumers(other.output()?).count() == 1,
                            "unfused Resize consumer"
                        );
                        variant = "add_resized";
                        extra = other.inputs()[0].clone();
                        fused.insert(other.output()?.to_string());
                    } else {
                        variant = "add";
                        extra = add.inputs()[1].clone();
                    }
                    fused.insert(add.output()?.to_string());
                    output = add.output()?;
                    aliases.insert(add.output()?.into(), node.output()?.into());
                }
                if let Some(relu) = net.consumers(output).find(|n| n.op_type() == "Relu") {
                    ensure!(
                        variant != "add_resized" && net.consumers(output).count() == 1,
                        "unsupported ReLU fusion"
                    );
                    variant = if variant == "plain" {
                        "relu"
                    } else {
                        "relu_add"
                    };
                    fused.insert(relu.output()?.to_string());
                    aliases.insert(relu.output()?.into(), node.output()?.into());
                }
                let stride = node.integers("strides", &[1, 1])[0] as usize;
                ensure!(
                    (taps == 9 && variant != "add_resized")
                        || (taps == 1
                            && stride == 1
                            && (variant == "plain" || variant == "add_resized")),
                    "unsupported convolution fusion"
                );
                let cp = if taps == 9 { align(ci, 8) } else { storage(ci) };
                let k = align(taps * cp, 32);
                let n = align(co, 64);
                emit16(
                    &mut weights,
                    name.clone(),
                    &pack(
                        &net.tensor(&node.inputs()[1])?.f64s()?,
                        co,
                        ci,
                        taps,
                        cp,
                        n,
                        k,
                    ),
                );
                let mut bias = vec![0.; n];
                bias[..co].copy_from_slice(&net.tensor(&node.inputs()[2])?.f64s()?);
                emit32(&mut weights, format!("{name}_b"), &bias);
                let s = net.shape(&node.inputs()[0])?;
                let out = net.shape(node.output()?)?;
                ops.push(Op {
                    kind: if taps == 9 { "conv" } else { "matmul" },
                    variant,
                    name,
                    src: node.inputs()[0].clone(),
                    dst: node.output()?.into(),
                    extra,
                    h: s[2],
                    w: s[3],
                    stride,
                    cin_pad: cp,
                    cin_stride: storage(ci),
                    k,
                    n,
                    ho: out[2],
                    wo: out[3],
                    // The 28-channel stem needs only half the usual WMMA tile.
                    tile: if co <= 32 && taps == 9 && variant == "relu" {
                        32
                    } else if n == 128 {
                        128
                    } else {
                        64
                    },
                    bytes: out[2] * out[3] * n * 2,
                    ..Default::default()
                });
            }
            "MaxPool" | "AveragePool" => {
                let s = net.shape(&node.inputs()[0])?;
                let out = net.shape(node.output()?)?;
                let c = storage(s[1]);
                ops.push(Op {
                    kind: "pool",
                    variant: if node.op_type() == "MaxPool" {
                        "max"
                    } else {
                        "mean"
                    },
                    name: node.name().to_owned(),
                    src: node.inputs()[0].clone(),
                    dst: node.output()?.into(),
                    h: s[2],
                    w: s[3],
                    ho: out[2],
                    wo: out[3],
                    cin_stride: c,
                    bytes: out[2] * out[3] * c * 2,
                    ..Default::default()
                });
            }
            "Relu" | "Add" => {
                ensure!(fused.contains(node.output()?), "unfused {}", node.op_type());
            }
            "Mul" | "Sigmoid" | "Transpose" | "Reshape" | "Resize" | "Shape" | "Gather"
            | "Unsqueeze" | "Slice" | "Concat" => {}
            other => anyhow::bail!("unsupported SCRFD operator {other}"),
        }
    }
    // Resize can precede the convolution that absorbs it, so check coverage
    // after all fusions have been identified. Shape operators are CPU-folded.
    for node in net.nodes() {
        if matches!(
            node.op_type(),
            "Mul" | "Sigmoid" | "Transpose" | "Reshape" | "Resize"
        ) {
            ensure!(fused.contains(node.output()?), "unfused {}", node.op_type());
        }
    }
    let mut ordered_heads: Vec<_> = heads.into_iter().collect();
    ordered_heads.sort_by_key(|(src, _)| std::cmp::Reverse(net.shapes[src][2]));
    ensure!(
        ordered_heads
            .iter()
            .zip([80, 40, 20])
            .all(|((src, _), size)| { net.shapes[src][2..] == [size, size] }),
        "expected head resolutions 80, 40 and 20"
    );
    let mut outputs = vec![];
    for (src, g) in ordered_heads {
        let s = net.shape(&src)?;
        let stride = 640 / s[2];
        ensure!(
            [8, 16, 32].contains(&stride) && s[2] == s[3],
            "invalid head resolution"
        );
        let name = format!("h{stride}");
        let ci = s[1];
        let cp = align(ci, 8);
        let k = align(9 * cp, 32);
        let mut w = vec![];
        let mut b = vec![];
        for co in [2, 8, 20] {
            let n = &net.nodes()[g[&co]];
            ensure!(
                net.tensor(&n.inputs()[1])?.shape()? == [co, ci, 3, 3]
                    && n.integers("strides", &[1, 1]) == [1, 1],
                "invalid head convolution"
            );
            let scale = head_scales[&g[&co]];
            // The old export multiplies float32 before conversion to half.
            w.extend(
                net.tensor(&n.inputs()[1])?
                    .f64s()?
                    .into_iter()
                    .map(|v| ((v as f32) * (scale as f32)) as f64),
            );
            b.extend(
                net.tensor(&n.inputs()[2])?
                    .f64s()?
                    .into_iter()
                    .map(|v| ((v as f32) * (scale as f32)) as f64),
            );
        }
        emit16(&mut weights, name.clone(), &pack(&w, 30, ci, 9, cp, 64, k));
        b.resize(64, 0.);
        emit32(&mut weights, format!("{name}_b"), &b);
        let dst = format!("head_{name}");
        outputs.push(dst.clone());
        ops.push(Op {
            kind: "conv",
            variant: "plain",
            name,
            src,
            dst,
            h: s[2],
            w: s[3],
            stride: 1,
            cin_pad: cp,
            cin_stride: storage(ci),
            k,
            n: 64,
            ho: s[2],
            wo: s[3],
            tile: 64,
            bytes: s[2] * s[3] * 64 * 2,
            ..Default::default()
        });
    }
    finish(ops, aliases, weights, &outputs)
}

fn validate_head(
    net: &Network,
    head: &Node,
    channels: usize,
    fused: &mut HashSet<String>,
) -> Result<(String, f64)> {
    let pipeline: &[&str] = match channels {
        2 => &["Transpose", "Reshape", "Sigmoid"],
        8 => &["Mul", "Transpose", "Reshape"],
        20 => &["Transpose", "Reshape"],
        _ => anyhow::bail!("unsupported head channels"),
    };
    let mut output = head.output()?;
    let mut scale = 1.;
    for expected in pipeline {
        let mut consumers = net.consumers(output);
        let node = consumers.next().context("incomplete SCRFD head pipeline")?;
        ensure!(
            consumers.next().is_none() && node.op_type() == *expected && node.inputs()[0] == output,
            "unsupported SCRFD head pipeline: expected only {expected} after {output}"
        );
        if node.op_type() == "Reshape" {
            ensure!(
                net.tensor(&node.inputs()[1])?.i64s()? == [-1, (channels / 2) as i64],
                "incorrect SCRFD head reshape"
            );
        } else if node.op_type() == "Mul" {
            scale = net.tensor(&node.inputs()[1])?.f64s()?[0];
        }
        fused.insert(node.output()?.to_string());
        output = node.output()?;
    }
    ensure!(
        net.consumers(output).next().is_none(),
        "unsupported operation after SCRFD head"
    );
    Ok((output.to_string(), scale))
}
