use super::*;
#[test]
#[ignore = "requires gfx1151 and Loom"]
fn resident_letterbox_matches_integer_bilinear_reference() -> Result<()> {
    use hrx::{
        image::{ImageOps, RgbResize},
        tensor::TensorDesc,
    };
    let context = ModelContext::new(Default::default())?;
    let ops = ImageOps::new(&context, 3)?;
    for (width, height) in [
        (640, 640),
        (37, 59),
        (1023, 679),
        (1, 3),
        (3, 1),
        (1280, 720),
        (4096, 7),
    ] {
        let desc =
            TensorDesc::new(DType::U8, vec![1, height, width, 3])?.with_layout(Layout::Nhwc)?;
        let resize = RgbResize::letterbox(height, width, SIZE, SIZE, false)?;
        for seed in [0, 91] {
            let rgb: Vec<_> = (0..width * height * 3)
                .map(|i| ((i * 37 + i / 11 + seed) % 256) as u8)
                .collect();
            let expected = detection::letterbox(Image {
                rgb: &rgb,
                width,
                height,
            })?
            .0;
            let before = context.runtime().statistics().downloaded_bytes;
            let resident = context.upload(desc.clone(), &rgb)?;
            let inference = ops.resize_rgb(&resident, resize)?;
            inference.completion().wait()?;
            assert_eq!(context.runtime().statistics().downloaded_bytes, before);
            let actual = inference.download()?.wait()?.remove(0);
            assert_eq!(actual.len(), expected.len());
            let mismatch = actual.iter().zip(&expected).position(|(a, b)| a != b);
            assert!(
                mismatch.is_none(),
                "{width}x{height} seed {seed}: first mismatch {mismatch:?}"
            );
        }
    }
    Ok(())
}

#[test]
#[ignore = "requires gfx1151"]
fn rgb_conversion_and_padding() -> Result<()> {
    use hrx::{
        loom::Specialization,
        model::{Command, Dispatch, ModelSession},
    };

    let mut engine = ModelSession::open_for(0, "gfx1151")?;
    let input = engine.allocate_shared(32 * 3)?;
    let output = engine.allocate_shared(32 * 8 * 2)?;
    let mut spec = Specialization::new("scrfd_hwc_u8_to_nhwc_f16");
    spec.set_config("scrfd.hwc_u8_to_nhwc_f16.size", "16");
    let kernels =
        unsafe { engine.compile(&[(include_str!("../kernels/hwc_u8_to_nhwc_f16.loom"), spec)])? };
    unsafe {
        engine.record(
            1,
            &[Command::Dispatch(Dispatch::indices(
                kernels[0],
                [2],
                [2, 1, 1],
                vec![input.read(), output.write()],
            ))],
        )?;
    }
    // Unequal channels catch accidental RGB/BGR reversal. Replay with changed
    // colors also checks that every channel, including padding, is overwritten.
    for seed in [0, 73] {
        let rgb: Vec<u8> = (0..32 * 3).map(|i| ((i * 37 + seed) % 256) as u8).collect();
        engine.upload(input, &rgb)?;
        engine.upload(output, &vec![0xff; output.len()])?;
        engine.replay(1)?;
        let mut bytes = vec![0u8; output.len()];
        engine.read_many(&mut [(output, &mut bytes)])?;
        for (p, pixel) in bytes.as_chunks::<16>().0.iter().enumerate() {
            for (c, value) in pixel.as_chunks::<2>().0.iter().enumerate() {
                let expected = if c < 3 {
                    (rgb[p * 3 + c] as f32 - 127.5) / 128.
                } else {
                    0.
                };
                assert_eq!(
                    half::f16::from_le_bytes([value[0], value[1]]),
                    half::f16::from_f32(expected),
                    "pixel {p}, channel {c}"
                );
            }
        }
    }
    Ok(())
}

#[test]
fn preprocessing_and_detection_edges() {
    assert!(
        detection::letterbox(Image {
            rgb: &[],
            width: 0,
            height: 0
        })
        .is_err()
    );
    let img = vec![127u8; 640 * 640 * 3];
    let (out, scale) = detection::letterbox(Image {
        rgb: &img,
        width: 640,
        height: 640,
    })
    .unwrap();
    assert_eq!(out, img);
    assert_eq!(scale, 1.);
    let d = Detection {
        bbox: [0., 0., 10., 10.],
        score: 0.9,
        landmarks: [[0.; 2]; 5],
    };
    let mut lower = d.clone();
    lower.score = 0.8;
    assert_eq!(
        detection::suppress(vec![lower, d], [640, 640], DetectionOptions::default()).len(),
        1
    );
    assert!(
        DetectionOptions {
            threshold: f32::NAN,
            ..Default::default()
        }
        .validate()
        .is_err()
    );
}
#[test]
#[ignore = "requires pretrained weights; CPU model import"]
fn importer_liveness() -> Result<()> {
    let p = model::load(std::path::Path::new(&model_path()?))?;
    assert_eq!(p.ops.len(), 57);
    assert_eq!(p.buffers.len(), 5);
    for l in p.ops {
        assert_ne!(l.src_buf, l.dst_buf);
        assert_ne!(l.extra_buf, l.dst_buf);
    }
    Ok(())
}
#[test]
#[ignore = "requires pretrained weights and gfx1151"]
fn native_reference_and_replay() -> Result<()> {
    let path = model_path()?;
    let model = Scrfd::load(
        &path,
        Options {
            device: 0,
            max_batch: 2,
        },
    )?;
    let canvas: Vec<u8> = (0..640 * 640 * 3)
        .map(|i| ((i * 7 + i / 101) % 256) as u8)
        .collect();
    model.detect_letterboxed(&canvas, &[1.], &[[640, 640]], Default::default())?;
    let read_heads = |canvases: &[u8]| -> Result<Vec<Vec<half::f16>>> {
        Ok(model
            .cnn
            .prepare(canvases.len() / (640 * 640 * 3))?
            .submit_host(&[canvases])?
            .download()?
            .wait()?
            .into_iter()
            .map(|bytes| {
                bytes
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|b| half::f16::from_le_bytes(*b))
                    .collect()
            })
            .collect())
    };
    let first = read_heads(&canvas)?;
    let mut blob = vec![0.; 3 * 640 * 640];
    for c in 0..3 {
        for y in 0..640 {
            for x in 0..640 {
                blob[(c * 640 + y) * 640 + x] =
                    (canvas[(y * 640 + x) * 3 + c] as f64 - 127.5) / 128.;
            }
        }
    }
    let expected = reference::forward(std::path::Path::new(&path), blob)?;
    for (level, size) in [80, 40, 20].into_iter().enumerate() {
        for (kind, width) in [1, 4, 10].into_iter().enumerate() {
            let data: Vec<f32> = expected[kind * 3 + level]
                .iter()
                .map(|v| *v as f32)
                .collect();
            let want = data.as_slice();
            let mut got = vec![];
            for p in 0..size * size {
                let row = &first[level][p * 64..][..64];
                for a in 0..2 {
                    for c in 0..width {
                        got.push(match kind {
                            0 => 1. / (1. + (-row[a].to_f32()).exp()),
                            1 => row[2 + 4 * a + c].to_f32(),
                            _ => row[10 + 10 * a + c].to_f32(),
                        });
                    }
                }
            }
            let max_error = got
                .iter()
                .zip(want)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            assert!(
                max_error < if kind == 0 { 0.02 } else { 0.15 },
                "head {kind}/{size} max error {max_error}"
            );
            let dot = got
                .iter()
                .zip(want)
                .map(|(a, b)| *a as f64 * *b as f64)
                .sum::<f64>();
            let norm = |a: &[f32]| a.iter().map(|x| (*x as f64).powi(2)).sum::<f64>();
            let cos = dot / (norm(&got) * norm(want)).sqrt();
            assert!(cos > 0.9999, "head {kind}/{size} cosine {cos}");
        }
    }
    let other = vec![33; canvas.len()];
    let mut pair = canvas.clone();
    pair.extend(&other);
    let pair_heads = read_heads(&pair)?;
    for (i, size) in [80, 40, 20].into_iter().enumerate() {
        assert_eq!(
            &first[i][..size * size * 64],
            &pair_heads[i][..size * size * 64]
        );
    }
    let replay_heads = read_heads(&canvas)?;
    for (i, size) in [80, 40, 20].into_iter().enumerate() {
        assert_eq!(
            &first[i][..size * size * 64],
            &replay_heads[i][..size * size * 64]
        );
    }
    Ok(())
}
#[test]
fn malformed_onnx_returns_errors() {
    use onnx_protobuf::{GraphProto, Message, ModelProto, NodeProto, ValueInfoProto};
    assert!(onnx::Network::from_bytes(&[], 112).is_err());
    assert!(onnx::Network::from_bytes(&[0x3a, 0xff], 112).is_err());
    let node = NodeProto {
        op_type: "Conv".into(),
        input: vec!["x".into()],
        output: vec!["y".into()],
        ..Default::default()
    };
    let graph = GraphProto {
        node: vec![node],
        input: vec![ValueInfoProto {
            name: "x".into(),
            ..Default::default()
        }],
        ..Default::default()
    };
    let model = ModelProto {
        graph: Some(graph).into(),
        ..Default::default()
    };
    assert!(onnx::Network::from_bytes(&model.write_to_bytes().unwrap(), 112).is_err());
}

#[test]
fn malformed_operator_ranks_return_errors() -> Result<()> {
    use onnx_protobuf::{
        AttributeProto, GraphProto, Message, ModelProto, NodeProto, ValueInfoProto,
    };
    let ints = |name: &str, values: &[i64]| AttributeProto {
        name: name.into(),
        ints: values.into(),
        ..Default::default()
    };
    let text = |name: &str, value: &str| AttributeProto {
        name: name.into(),
        s: value.as_bytes().into(),
        ..Default::default()
    };
    for op in ["MaxPool", "AveragePool", "Resize", "Transpose"] {
        let (inputs, attribute) = match op {
            "Resize" => (
                vec!["flat", "", "", "sizes"],
                vec![
                    text("mode", "nearest"),
                    text("coordinate_transformation_mode", "asymmetric"),
                    text("nearest_mode", "floor"),
                ],
            ),
            "Transpose" => (vec!["flat"], vec![ints("perm", &[2, 3, 0, 1])]),
            _ => (
                vec!["flat"],
                vec![ints("kernel_shape", &[2, 2]), ints("strides", &[2, 2])],
            ),
        };
        let graph = GraphProto {
            input: vec![ValueInfoProto {
                name: "x".into(),
                ..Default::default()
            }],
            node: vec![
                NodeProto {
                    op_type: "Flatten".into(),
                    input: vec!["x".into()],
                    output: vec!["flat".into()],
                    ..Default::default()
                },
                NodeProto {
                    op_type: op.into(),
                    input: inputs.into_iter().map(String::from).collect(),
                    output: vec!["y".into()],
                    attribute,
                    ..Default::default()
                },
            ],
            ..Default::default()
        };
        let model = ModelProto {
            graph: Some(graph).into(),
            ..Default::default()
        };
        // A panic fails the test; require the error to identify the rank issue.
        let error = onnx::Network::from_bytes(&model.write_to_bytes()?, 640)
            .err()
            .expect("invalid rank accepted");
        assert!(
            error
                .to_string()
                .contains("expected rank-4 input, found rank-2"),
            "{op}: {error}"
        );
    }
    Ok(())
}

#[test]
#[ignore = "requires pretrained weights; CPU model import"]
fn importer_rejects_modified_head_pipelines() -> Result<()> {
    use onnx_protobuf::{Message, ModelProto, NodeProto, TensorProto};
    let original = ModelProto::parse_from_bytes(&std::fs::read(model_path()?)?)?;
    let dir = tempfile::tempdir()?;
    let path = dir.path().join("modified.onnx");
    for mutation in [
        "scale_scores",
        "replace_sigmoid",
        "wrong_reshape",
        "wrong_outputs",
        "unused_operation",
    ] {
        let mut modified = original.clone();
        let graph = modified.graph.as_mut().unwrap();
        graph.initializer.push(TensorProto {
            name: "review_scale".into(),
            dims: vec![1],
            data_type: 1,
            float_data: vec![0.5],
            ..Default::default()
        });
        let sigmoid = graph
            .node
            .iter()
            .position(|n| n.op_type == "Sigmoid")
            .unwrap();
        match mutation {
            "scale_scores" => {
                let output = graph.node[sigmoid].output[0].clone();
                graph.node[sigmoid].output[0] = "unscaled_scores".into();
                graph.node.insert(
                    sigmoid + 1,
                    NodeProto {
                        op_type: "Mul".into(),
                        input: vec!["unscaled_scores".into(), "review_scale".into()],
                        output: vec![output],
                        ..Default::default()
                    },
                );
            }
            "replace_sigmoid" => {
                graph.node[sigmoid].op_type = "Mul".into();
                graph.node[sigmoid].input.push("review_scale".into());
            }
            "wrong_reshape" => {
                let reshape = graph
                    .node
                    .iter_mut()
                    .find(|n| n.op_type == "Reshape")
                    .unwrap();
                reshape.input[1] = "review_shape".into();
                graph.initializer.push(TensorProto {
                    name: "review_shape".into(),
                    dims: vec![2],
                    data_type: 7,
                    int64_data: vec![-1, 4],
                    ..Default::default()
                });
            }
            "wrong_outputs" => graph.output[0].name = graph.input[0].name.clone(),
            _ => graph.node.push(NodeProto {
                op_type: "Mul".into(),
                input: vec![graph.input[0].name.clone(), "review_scale".into()],
                output: vec!["unused".into()],
                ..Default::default()
            }),
        }
        std::fs::write(&path, modified.write_to_bytes()?)?;
        // Each variant is shape-valid ONNX; rejection must happen in the SCRFD importer.
        onnx::Network::load(&path, 640)?;
        let error = model::load(&path)
            .err()
            .expect("modified pipeline accepted");
        assert!(
            error.to_string().contains("head") || error.to_string().contains("unfused"),
            "{mutation}: {error}"
        );
    }
    Ok(())
}
#[test]
#[ignore = "requires pretrained weights and gfx1151"]
fn insightface_fixture() -> Result<()> {
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("../tests/fixtures/t1_insightface.json"))?;
    let expected: Vec<[f32; 5]> = serde_json::from_value(fixture["det"].clone())?;
    let kps: Vec<[[f32; 2]; 5]> = serde_json::from_value(fixture["kps"].clone())?;
    let image = image::load_from_memory(include_bytes!("../tests/fixtures/t1.png"))?.to_rgb8();
    let (w, h) = image.dimensions();
    let rgb = image.into_raw();
    let model = Scrfd::load(
        model_path()?,
        Options {
            device: 0,
            max_batch: 2,
        },
    )?;
    let image = Image {
        rgb: &rgb,
        width: w as usize,
        height: h as usize,
    };
    let got = model.detect(image, Default::default())?;
    assert_eq!(got.len(), expected.len());
    for ((a, b), k) in got.iter().zip(&expected).zip(&kps) {
        let aa = a.bbox;
        let inter = (aa[2].min(b[2]) - aa[0].max(b[0])).max(0.)
            * (aa[3].min(b[3]) - aa[1].max(b[1])).max(0.);
        let union = (aa[2] - aa[0]) * (aa[3] - aa[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter;
        assert!(inter / union > 0.99);
        assert!((a.score - b[4]).abs() < 0.01);
        for (x, y) in a.landmarks.iter().flatten().zip(k.iter().flatten()) {
            assert!((x - y).abs() < 1.);
        }
    }
    let batched = model.detect_batch(&[image, image], Default::default())?;
    assert_eq!(
        serde_json::to_value(&got)?,
        serde_json::to_value(&batched[0])?
    );
    assert_eq!(
        serde_json::to_value(&got)?,
        serde_json::to_value(&batched[1])?
    );
    Ok(())
}

fn model_path() -> Result<std::path::PathBuf> {
    match std::env::var_os("SCRFD_MODEL") {
        Some(path) => Ok(path.into()),
        None => hub::weights(false),
    }
}

#[test]
#[ignore = "requires gfx1151 and Loom compiler"]
fn gpu_decode_matches_stable_cpu_oracle_and_errors() -> Result<()> {
    use hrx::tensor::TensorDesc;
    let context = ModelContext::new(Default::default())?;
    let decoder = postprocess::Postprocess::new(&context)?;
    let mut heads: Vec<Vec<half::f16>> = [80, 40, 20]
        .map(|size| {
            let mut data = vec![half::f16::ZERO; 2 * size * size * 64];
            for row in data.as_chunks_mut::<64>().0 {
                row[..2].fill(half::f16::from_f32(-10.));
            }
            data
        })
        .into();
    for (level, pixel, anchor, logit, distance) in [
        (0, 0, 0, 0., 1.),
        (0, 0, 1, 0., 1.),
        (0, 50, 0, 1., 2.),
        (1, 50, 0, 1., 3.),
        (2, 70, 1, 2., 1.),
    ] {
        let row = &mut heads[level][pixel * 64..][..64];
        row[anchor] = half::f16::from_f32(logit);
        row[2 + 4 * anchor..6 + 4 * anchor].fill(half::f16::from_f32(distance));
        row[10 + 10 * anchor..20 + 10 * anchor].fill(half::f16::from_f32(anchor as f32));
    }
    let upload = |heads: &[Vec<half::f16>]| -> Result<Vec<DeviceTensor>> {
        heads
            .iter()
            .zip([80, 40, 20])
            .map(|(data, size)| {
                Ok(context.upload(
                    TensorDesc::new(DType::F16, vec![2, size, size, 64])?
                        .with_layout(Layout::Nhwc)?,
                    bytemuck::cast_slice(data),
                )?)
            })
            .collect()
    };
    for options in [
        DetectionOptions::default(),
        DetectionOptions {
            max_detections: 2,
            ranking: Ranking::Area,
            ..Default::default()
        },
        DetectionOptions {
            max_detections: 2,
            ranking: Ranking::AreaAndCenter,
            ..Default::default()
        },
        DetectionOptions {
            threshold: 0.7310586,
            ..Default::default()
        },
        DetectionOptions {
            max_candidates: 5,
            max_detections: 1,
            ..Default::default()
        },
    ] {
        let input = upload(&heads)?;
        let before = context.runtime().statistics().downloaded_bytes;
        let detections = decoder.submit(&input, &[0.7, 1.], &[[901, 703], [640, 640]], options)?;
        let candidate_bytes = detections.candidate_counts().iter().sum::<usize>() as u64 * 20;
        let got = detections.wait()?;
        let downloaded = context.runtime().statistics().downloaded_bytes - before;
        assert_eq!(
            downloaded,
            16 + candidate_bytes + got.iter().map(|r| r.len() as u64 * 64).sum::<u64>()
        );
        for b in 0..2 {
            let want = detection::decode(
                &heads,
                b,
                [0.7, 1.][b],
                [[901, 703], [640, 640]][b],
                options,
            )?;
            assert_eq!(got[b].len(), want.len());
            for (a, b) in got[b].iter().zip(&want) {
                assert_eq!(a.score, b.score);
                for (a, b) in a
                    .bbox
                    .iter()
                    .chain(a.landmarks.iter().flatten())
                    .zip(b.bbox.iter().chain(b.landmarks.iter().flatten()))
                {
                    assert!((a - b).abs() < 2e-4, "coordinate {a} vs {b}");
                }
            }
        }
    }
    let overflow = DetectionOptions {
        max_candidates: 2,
        ..Default::default()
    };
    assert!(
        decoder
            .submit(&upload(&heads)?, &[1.; 2], &[[640, 640]; 2], overflow)
            .and_then(postprocess::Detections::wait)
            .unwrap_err()
            .to_string()
            .contains("capacity")
    );
    heads[0][0] = half::f16::INFINITY;
    assert!(
        decoder
            .submit(
                &upload(&heads)?,
                &[1.; 2],
                &[[640, 640]; 2],
                Default::default()
            )
            .and_then(postprocess::Detections::wait)
            .unwrap_err()
            .to_string()
            .contains("logit")
    );
    heads[0][0] = half::f16::from_f32(1.);
    heads[0][2] = half::f16::NAN;
    assert!(
        decoder
            .submit(
                &upload(&heads)?,
                &[1.; 2],
                &[[640, 640]; 2],
                Default::default()
            )
            .and_then(postprocess::Detections::wait)
            .unwrap_err()
            .to_string()
            .contains("output")
    );
    Ok(())
}

/// Synthetic dense heads exercise the worst-case score selection, suppression,
/// and ranking paths without requiring a model checkpoint. Timings include
/// compact readback but exclude head upload; preparation is reported separately.
#[test]
#[ignore = "requires gfx1151 and Loom compiler; dense postprocessing qualification"]
fn dense_decode_qualification() -> Result<()> {
    use hrx::tensor::TensorDesc;
    use std::time::Instant;

    let context = ModelContext::new(Default::default())?;
    let decoder = postprocess::Postprocess::new(&context)?;
    for count in [256, 4096] {
        for (label, distance, max_detections) in [
            ("disjoint", 0., 0),
            ("overlapping", 40., 0),
            ("ranked", 0., 16),
        ] {
            let mut heads: Vec<Vec<half::f16>> = [80, 40, 20]
                .map(|size| {
                    let mut head = vec![half::f16::ZERO; size * size * 64];
                    for row in head.as_chunks_mut::<64>().0 {
                        row[..2].fill(half::f16::from_f32(-10.));
                    }
                    head
                })
                .into();
            for (i, row) in heads[0]
                .as_chunks_mut::<64>()
                .0
                .iter_mut()
                .take(count)
                .enumerate()
            {
                // Repeated scores verify stable ties independently of ranking.
                row[0] = half::f16::from_f32((i % 13) as f32 / 8.);
                row[2..6].fill(half::f16::from_f32(distance));
            }
            let options = DetectionOptions {
                max_detections,
                ..Default::default()
            };
            let cpu_start = Instant::now();
            let expected = detection::decode(&heads, 0, 1., [640, 640], options)?;
            let cpu = cpu_start.elapsed();
            let inputs: Vec<_> = heads
                .iter()
                .zip([80, 40, 20])
                .map(|(head, size)| {
                    context.upload(
                        TensorDesc::new(DType::F16, vec![1, size, size, 64])?
                            .with_layout(Layout::Nhwc)?,
                        bytemuck::cast_slice(head),
                    )
                })
                .collect::<hrx::Result<_>>()?;
            let mut elapsed = Vec::new();
            let mut warmed = None;
            for replay in 0..4 {
                let before = context.runtime().statistics();
                let start = Instant::now();
                let actual = decoder
                    .submit(&inputs, &[1.], &[[640, 640]], options)?
                    .wait()?;
                elapsed.push(start.elapsed());
                assert_eq!(actual[0].len(), expected.len(), "{label}/{count}");
                for (a, b) in actual[0].iter().zip(&expected) {
                    assert_eq!(a.score, b.score, "{label}/{count}");
                    assert_eq!(a.bbox, b.bbox, "{label}/{count}");
                    assert_eq!(a.landmarks, b.landmarks, "{label}/{count}");
                }
                let after = context.runtime().statistics();
                assert_eq!(
                    after.downloaded_bytes - before.downloaded_bytes,
                    8 + count as u64 * 20 + expected.len() as u64 * 64
                );
                if replay == 1 {
                    warmed = Some(after);
                } else if replay > 1 {
                    let baseline = warmed.as_ref().unwrap();
                    // Completion wakes callers before the worker drops its last
                    // staging reference. Check eventual release, not a racy
                    // snapshot taken while that destructor is still running.
                    let deadline = Instant::now() + std::time::Duration::from_secs(1);
                    while context.runtime().statistics().live_bytes > baseline.live_bytes {
                        assert!(Instant::now() < deadline, "warm storage was not released");
                        std::thread::yield_now();
                    }
                    assert_eq!(
                        after.native_graphs_prepared,
                        baseline.native_graphs_prepared
                    );
                    assert_eq!(after.copy_streams_created, baseline.copy_streams_created);
                }
            }
            eprintln!(
                "dense {label}/{count}: kept={}, CPU={cpu:?}, hybrid prepare+first={:?}, warm={:?}",
                expected.len(),
                elapsed[0],
                &elapsed[1..]
            );
        }
    }
    Ok(())
}
