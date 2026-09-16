//! Test-only float64 evaluation of the unfused ONNX operators.
use crate::onnx::Network;
use anyhow::{Context, Result};
use std::{collections::HashMap, path::Path};
pub fn forward(path: &Path, input: Vec<f64>) -> Result<Vec<Vec<f64>>> {
    let net = Network::load(path, 640)?;
    let mut values = HashMap::from([(net.inputs()[0].clone(), input)]);
    for node in net.nodes() {
        let get = |s: &str| {
            values
                .get(s)
                .with_context(|| format!("reference missing {s}"))
        };
        let output = match node.op_type() {
            "Conv" => {
                let src = get(&node.inputs()[0])?;
                let s = net.shape(&node.inputs()[0])?;
                let ws = net.tensor(&node.inputs()[1])?.shape()?;
                let w = net.tensor(&node.inputs()[1])?.f64s()?;
                let bias = net.tensor(&node.inputs()[2])?.f64s()?;
                let sh = net.shape(node.output()?)?;
                let (co, ci, kh, kw) = (ws[0], ws[1], ws[2], ws[3]);
                let (ho, wo) = (sh[2], sh[3]);
                let m = ho * wo;
                let k = ci * kh * kw;
                let stride = node.integers("strides", &[1, 1])[0] as usize;
                let pad = node.integers("pads", &[0, 0, 0, 0])[0] as isize;
                let mut columns = vec![0.; k * m];
                for c in 0..ci {
                    for dy in 0..kh {
                        for dx in 0..kw {
                            for y in 0..ho {
                                for x in 0..wo {
                                    let sy = (y * stride + dy) as isize - pad;
                                    let sx = (x * stride + dx) as isize - pad;
                                    if sy >= 0
                                        && sx >= 0
                                        && (sy as usize) < s[2]
                                        && (sx as usize) < s[3]
                                    {
                                        columns[((c * kh + dy) * kw + dx) * m + y * wo + x] =
                                            src[(c * s[2] + sy as usize) * s[3] + sx as usize];
                                    }
                                }
                            }
                        }
                    }
                }
                let mut out = vec![0.; co * m];
                // Validated dimensions above describe dense, disjoint matrices.
                unsafe {
                    matrixmultiply::dgemm(
                        co,
                        k,
                        m,
                        1.,
                        w.as_ptr(),
                        k as isize,
                        1,
                        columns.as_ptr(),
                        m as isize,
                        1,
                        0.,
                        out.as_mut_ptr(),
                        m as isize,
                        1,
                    );
                }
                for (row, b) in out.chunks_mut(m).zip(bias) {
                    for v in row {
                        *v += b;
                    }
                }
                out
            }
            "Relu" => get(&node.inputs()[0])?.iter().map(|v| v.max(0.)).collect(),
            "Sigmoid" => get(&node.inputs()[0])?
                .iter()
                .map(|v| 1. / (1. + (-v).exp()))
                .collect(),
            "Add" => get(&node.inputs()[0])?
                .iter()
                .zip(get(&node.inputs()[1])?)
                .map(|(a, b)| a + b)
                .collect(),
            "Mul" => {
                let scale = net.tensor(&node.inputs()[1])?.f64s()?[0];
                get(&node.inputs()[0])?.iter().map(|v| v * scale).collect()
            }
            "MaxPool" | "AveragePool" => {
                let src = get(&node.inputs()[0])?;
                let s = net.shape(&node.inputs()[0])?;
                let (h, w) = (s[2] / 2, s[3] / 2);
                let mut out = vec![0.; s[1] * h * w];
                for c in 0..s[1] {
                    for y in 0..h {
                        for x in 0..w {
                            let vals = [
                                src[(c * s[2] + y * 2) * s[3] + x * 2],
                                src[(c * s[2] + y * 2) * s[3] + x * 2 + 1],
                                src[(c * s[2] + y * 2 + 1) * s[3] + x * 2],
                                src[(c * s[2] + y * 2 + 1) * s[3] + x * 2 + 1],
                            ];
                            out[(c * h + y) * w + x] = if node.op_type() == "MaxPool" {
                                vals.into_iter().fold(f64::NEG_INFINITY, f64::max)
                            } else {
                                vals.iter().sum::<f64>() / 4.
                            };
                        }
                    }
                }
                out
            }
            "Resize" => {
                let src = get(&node.inputs()[0])?;
                let s = net.shape(&node.inputs()[0])?;
                let (h, w) = (s[2] * 2, s[3] * 2);
                let mut out = vec![0.; s[1] * h * w];
                for c in 0..s[1] {
                    for y in 0..h {
                        for x in 0..w {
                            out[(c * h + y) * w + x] = src[(c * s[2] + y / 2) * s[3] + x / 2];
                        }
                    }
                }
                out
            }
            "Transpose" => {
                let src = get(&node.inputs()[0])?;
                let s = net.shape(&node.inputs()[0])?;
                let m = s[2] * s[3];
                let mut out = vec![0.; src.len()];
                for c in 0..s[1] {
                    for p in 0..m {
                        out[p * s[1] + c] = src[c * m + p];
                    }
                }
                out
            }
            "Reshape" => get(&node.inputs()[0])?.clone(),
            "Shape" | "Gather" | "Unsqueeze" | "Slice" | "Concat" => continue,
            other => anyhow::bail!("unsupported reference operator {other}"),
        };
        values.insert(node.output()?.to_owned(), output);
    }
    net.outputs()
        .iter()
        .map(|v| values.remove(v).context("missing reference output"))
        .collect()
}
