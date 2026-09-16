//! Validation and fixed-shape resolution for the supported ONNX model.
use anyhow::{Context, Result, ensure};
use hrx::artifacts::onnx::Model;
pub(crate) use hrx::artifacts::onnx::{Node, Tensor};
use std::{collections::HashMap, path::Path};
pub struct Network {
    model: Model,
    pub shapes: HashMap<String, Vec<usize>>,
}
impl Network {
    pub fn tensor(&self, name: &str) -> Result<&Tensor> {
        Ok(self.model.initializer(name)?)
    }
    pub fn producer(&self, name: &str) -> Option<&Node> {
        self.model.producer(name)
    }
    pub fn consumers<'a>(&'a self, name: &'a str) -> impl Iterator<Item = &'a Node> {
        self.model
            .consumers(name)
            .filter(|node| node.op_type() != "Shape")
    }
    pub fn nodes(&self) -> &[Node] {
        self.model.nodes()
    }
    pub fn outputs(&self) -> &[String] {
        self.model.outputs()
    }
    pub fn inputs(&self) -> &[String] {
        self.model.inputs()
    }
    pub fn shape(&self, name: &str) -> Result<&[usize]> {
        Ok(self
            .shapes
            .get(name)
            .with_context(|| format!("missing shape {name}"))?)
    }
    pub fn load(path: &Path, size: usize) -> Result<Self> {
        Self::from_model(Model::load(path)?, size)
    }
    #[cfg(test)]
    pub(crate) fn from_bytes(bytes: &[u8], size: usize) -> Result<Self> {
        Self::from_model(Model::from_bytes(bytes)?, size)
    }
    fn from_model(model: Model, size: usize) -> Result<Self> {
        ensure!(
            model.inputs().len() == 1 && !model.nodes().is_empty(),
            "expected one model input"
        );
        let mut net = Self {
            shapes: HashMap::from([(model.inputs()[0].clone(), vec![1, 3, size, size])]),
            model,
        };
        let mut folded: HashMap<String, Vec<i64>> = HashMap::new();
        for n in net.model.nodes() {
            ensure!(
                n.outputs().len() == 1
                    && !n.inputs().is_empty()
                    && (n.domain().is_empty() || n.domain() == "ai.onnx"),
                "unsupported node {}",
                n.name()
            );
            ensure!(
                !net.shapes.contains_key(n.output()?) && !folded.contains_key(n.output()?),
                "duplicate output"
            );
            let (min, max, allowed): (usize, usize, &[&str]) = match n.op_type() {
                "Conv" => (
                    3,
                    3,
                    &[
                        "pads",
                        "kernel_shape",
                        "dilations",
                        "strides",
                        "group",
                        "auto_pad",
                    ],
                ),
                "BatchNormalization" => (5, 5, &["epsilon", "momentum", "training_mode"]),
                "PRelu" | "Add" | "Mul" => (2, 2, &[]),
                "Relu" | "Sigmoid" | "Shape" => (1, 1, &[]),
                "Flatten" => (1, 1, &["axis"]),
                "Gemm" => (3, 3, &["alpha", "beta", "transA", "transB"]),
                "MaxPool" | "AveragePool" => (
                    1,
                    1,
                    &[
                        "pads",
                        "ceil_mode",
                        "strides",
                        "kernel_shape",
                        "count_include_pad",
                    ],
                ),
                "Gather" => (2, 2, &["axis"]),
                "Unsqueeze" => (1, 1, &["axes"]),
                "Slice" => (3, 5, &[]),
                "Concat" => (1, 8, &["axis"]),
                "Resize" => (
                    4,
                    4,
                    &[
                        "coordinate_transformation_mode",
                        "cubic_coeff_a",
                        "nearest_mode",
                        "mode",
                    ],
                ),
                "Transpose" => (1, 1, &["perm"]),
                "Reshape" => (2, 2, &[]),
                other => anyhow::bail!("unsupported ONNX operator {other}"),
            };
            ensure!(
                (min..=max).contains(&n.inputs().len()),
                "{}: invalid input count",
                n.op_type()
            );
            n.validate_attributes(allowed)?;
            let input = || net.shape(&n.inputs()[0]);
            let ranked_input = |rank: usize| -> Result<&[usize]> {
                let s = input()?;
                ensure!(
                    s.len() == rank,
                    "{}: expected rank-{rank} input, found rank-{}",
                    n.op_type(),
                    s.len()
                );
                Ok(s)
            };
            let shape = match n.op_type() {
                "Conv" => {
                    ensure!(n.inputs().len() == 3, "convolution must include bias");
                    let w = net.tensor(&n.inputs()[1])?.shape()?;
                    let s = ranked_input(4)?;
                    ensure!(
                        w.len() == 4 && s.len() == 4 && w[1] == s[1] && w[2] == w[3],
                        "invalid convolution shape"
                    );
                    ensure!(
                        w[0] <= 512 && w[1] <= 512 && s[2] <= 640 && s[3] <= 640,
                        "unsupported convolution extent"
                    );
                    ensure!(
                        n.integers("kernel_shape", &[w[2] as i64, w[3] as i64])
                            == [w[2] as i64, w[3] as i64],
                        "convolution kernel shape mismatch"
                    );

                    let strides = n.integers("strides", &[1, 1]);
                    let pads = n.integers("pads", &[0, 0, 0, 0]);
                    ensure!(
                        strides.len() == 2
                            && strides[0] == strides[1]
                            && (1..=2).contains(&strides[0]),
                        "invalid convolution stride"
                    );
                    ensure!(
                        pads.len() == 4
                            && pads.iter().all(|p| *p == pads[0])
                            && ((w[2] == 3 && pads[0] == 1) || (w[2] == 1 && pads[0] == 0)),
                        "invalid convolution padding"
                    );
                    ensure!(
                        n.integer("group", 1) == 1
                            && n.integers("dilations", &[1, 1]) == [1, 1]
                            && n.text("auto_pad", "NOTSET")? == "NOTSET",
                        "unsupported convolution attributes"
                    );
                    ensure!(
                        net.tensor(&n.inputs()[2])?.shape()? == [w[0]],
                        "invalid convolution bias"
                    );
                    vec![
                        1,
                        w[0],
                        (s[2] + 2 * pads[0] as usize - w[2]) / strides[0] as usize + 1,
                        (s[3] + 2 * pads[0] as usize - w[3]) / strides[0] as usize + 1,
                    ]
                }
                "BatchNormalization" => {
                    ensure!(
                        n.inputs().len() == 5 && n.integer("training_mode", 0) == 0,
                        "unsupported BatchNorm"
                    );
                    let s = input()?.to_vec();
                    ensure!(
                        s.len() >= 2,
                        "BatchNormalization requires a channel dimension"
                    );
                    for t in &n.inputs()[1..] {
                        ensure!(net.tensor(t)?.shape()? == [s[1]], "invalid BatchNorm shape");
                    }
                    s
                }
                "PRelu" => {
                    ensure!(n.inputs().len() == 2, "invalid PRelu");
                    let s = input()?;
                    ensure!(s.len() >= 2, "PRelu requires a channel dimension");
                    ensure!(
                        net.tensor(&n.inputs()[1])?.count()? == s[1],
                        "invalid slope"
                    );
                    s.to_vec()
                }
                "Relu" | "Sigmoid" => input()?.to_vec(),
                "Add" => {
                    ensure!(
                        n.inputs().len() == 2 && input()? == net.shape(&n.inputs()[1])?,
                        "invalid Add"
                    );
                    input()?.to_vec()
                }
                "Mul" => {
                    ensure!(
                        n.inputs().len() == 2 && net.tensor(&n.inputs()[1])?.f64s()?.len() == 1,
                        "expected scalar Mul"
                    );
                    input()?.to_vec()
                }
                "MaxPool" | "AveragePool" => {
                    ensure!(
                        n.integers("kernel_shape", &[]) == [2, 2]
                            && n.integers("strides", &[1, 1]) == [2, 2]
                            && n.integers("pads", &[0, 0, 0, 0]) == [0, 0, 0, 0]
                            && [0, 1].contains(&n.integer("ceil_mode", 0)),
                        "unsupported pool"
                    );
                    let s = ranked_input(4)?;
                    ensure!(
                        s[2].is_multiple_of(2) && s[3].is_multiple_of(2),
                        "pool requires even spatial dimensions"
                    );
                    vec![1, s[1], s[2] / 2, s[3] / 2]
                }
                "Flatten" => {
                    ensure!(n.integer("axis", 1) == 1, "unsupported flatten");
                    let s = input()?;
                    ensure!(!s.is_empty(), "Flatten requires a non-scalar input");
                    vec![1, s[1..].iter().product()]
                }
                "Gemm" => {
                    ensure!(
                        n.inputs().len() == 3
                            && n.integer("transA", 0) == 0
                            && n.integer("transB", 0) == 1
                            && n.float("alpha", 1.) == 1.
                            && n.float("beta", 1.) == 1.,
                        "unsupported Gemm"
                    );
                    let w = net.tensor(&n.inputs()[1])?.shape()?;
                    let s = ranked_input(2)?;
                    ensure!(
                        w.len() == 2
                            && s[1] == w[1]
                            && net.tensor(&n.inputs()[2])?.shape()? == [w[0]],
                        "invalid Gemm shape"
                    );
                    vec![1, w[0]]
                }
                "Shape" => {
                    folded.insert(
                        n.output()?.into(),
                        input()?.iter().map(|x| *x as i64).collect(),
                    );
                    continue;
                }
                "Gather" => {
                    ensure!(n.integer("axis", 0) == 0, "unsupported Gather");
                    let v = folded.get(&n.inputs()[0]).context("unresolved Gather")?;
                    let ix = net.tensor(&n.inputs()[1])?.i64s()?;
                    let result = ix
                        .iter()
                        .map(|i| v.get(*i as usize).copied().context("invalid Gather index"))
                        .collect::<Result<_>>()?;
                    folded.insert(n.output()?.into(), result);
                    continue;
                }
                "Unsqueeze" => {
                    ensure!(n.integers("axes", &[]) == [0], "unsupported Unsqueeze axes");
                    let v = folded
                        .get(&n.inputs()[0])
                        .context("unresolved Unsqueeze")?
                        .clone();
                    folded.insert(n.output()?.into(), v);
                    continue;
                }
                "Slice" => {
                    if n.inputs().len() > 3 {
                        ensure!(
                            net.tensor(&n.inputs()[3])?.i64s()? == [0],
                            "unsupported Slice axis"
                        );
                    }
                    if n.inputs().len() > 4 {
                        ensure!(
                            net.tensor(&n.inputs()[4])?.i64s()? == [1],
                            "unsupported Slice step"
                        );
                    }
                    let v = folded.get(&n.inputs()[0]).context("unresolved Slice")?;
                    let st = net.tensor(&n.inputs()[1])?.i64s()?;
                    let en = net.tensor(&n.inputs()[2])?.i64s()?;
                    ensure!(
                        st.len() == 1 && en.len() == 1 && st[0] >= 0 && en[0] >= st[0],
                        "invalid Slice"
                    );
                    let end = (en[0] as usize).min(v.len());
                    let result = v
                        .get(st[0] as usize..end)
                        .context("invalid slice span")?
                        .to_vec();
                    folded.insert(n.output()?.into(), result);
                    continue;
                }
                "Concat" => {
                    ensure!(n.integer("axis", 0) == 0, "unsupported shape Concat");
                    let mut v = vec![];
                    for i in n.inputs() {
                        v.extend(folded.get(i).context("unresolved Concat")?);
                    }
                    folded.insert(n.output()?.into(), v);
                    continue;
                }
                "Resize" => {
                    ensure!(
                        n.text("mode", "")? == "nearest"
                            && n.text("coordinate_transformation_mode", "")? == "asymmetric"
                            && n.text("nearest_mode", "")? == "floor"
                            && n.inputs().len() == 4,
                        "unsupported Resize"
                    );
                    let s = ranked_input(4)?;
                    let out = vec![1, s[1], s[2] * 2, s[3] * 2];
                    ensure!(
                        folded.get(&n.inputs()[3])
                            == Some(&out.iter().map(|x| *x as i64).collect()),
                        "Resize must double spatial dimensions"
                    );
                    out
                }
                "Transpose" => {
                    ensure!(
                        n.integers("perm", &[]) == [2, 3, 0, 1],
                        "unsupported transpose"
                    );
                    let s = ranked_input(4)?;
                    vec![s[2], s[3], s[0], s[1]]
                }
                "Reshape" => {
                    ensure!(n.inputs().len() == 2, "invalid reshape");
                    let target = net.tensor(&n.inputs()[1])?.i64s()?;
                    ensure!(
                        target.len() == 2 && target[0] == -1 && [1, 4, 10].contains(&target[1]),
                        "unsupported head reshape"
                    );
                    let total: usize = input()?.iter().product();
                    ensure!(
                        total.is_multiple_of(target[1] as usize),
                        "invalid reshape size"
                    );
                    vec![total / target[1] as usize, target[1] as usize]
                }
                other => anyhow::bail!("unsupported ONNX operator {other}"),
            };
            net.shapes.insert(n.output()?.into(), shape);
        }
        for out in net.outputs() {
            net.shape(out)?;
        }
        Ok(net)
    }
}
