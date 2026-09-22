//! Resident-canvas detection latency, including compact CPU selection and readback.
use anyhow::{Result, ensure};
use hrx::tensor::{DType, Layout, TensorDesc};
use scrfd_hrx::{DetectionOptions, Image, Options, Scrfd};
use std::time::Instant;

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    let batch: usize = args.get(1).map(|v| v.parse()).transpose()?.unwrap_or(1);
    ensure!((1..=64).contains(&batch), "batch must be 1..=64");
    let rgb = image::load_from_memory(include_bytes!("../tests/fixtures/t1.png"))?.to_rgb8();
    let image = Image {
        rgb: rgb.as_raw(),
        width: rgb.width() as usize,
        height: rgb.height() as usize,
    };
    let model = Scrfd::load(
        scrfd_hrx::hub::weights(true)?,
        Options {
            max_batch: batch,
            ..Default::default()
        },
    )?;
    let options = DetectionOptions::default();
    let expected = serde_json::to_vec(&model.detect_batch(&vec![image; batch], options)?)?;
    let pixels = model.context().upload(
        TensorDesc::new(DType::U8, vec![batch, image.height, image.width, 3])?
            .with_layout(Layout::Nhwc)?,
        &image.rgb.repeat(batch),
    )?;
    let (letterboxed, scale) = model.letterbox(&pixels)?;
    let resident = letterboxed.outputs()[0].clone();
    let scales = vec![scale; batch];
    let shapes = vec![[image.width, image.height]; batch];
    let run = || model.submit(&resident, &scales, &shapes, options)?.wait();
    for _ in 0..5 {
        ensure!(
            serde_json::to_vec(&run()?)? == expected,
            "warmup differs from host entry point"
        );
    }
    let before = model.context().runtime().statistics();
    let mut timings = Vec::new();
    for _ in 0..41 {
        let start = Instant::now();
        let output = run()?;
        timings.push(start.elapsed().as_secs_f64() * 1000.);
        ensure!(
            serde_json::to_vec(&output)? == expected,
            "replay differs from host entry point"
        );
    }
    let after = model.context().runtime().statistics();
    let mut sorted = timings.clone();
    sorted.sort_by(f64::total_cmp);
    println!(
        "{}",
        serde_json::json!({"batch":batch,"median_ms":sorted[20],"samples_ms":timings,"output_sha256":hrx::bundle::digest(&expected),"submissions":after.submissions-before.submissions,"copied_bytes":after.copied_bytes-before.copied_bytes,"peak_bytes":after.peak_bytes})
    );
    Ok(())
}
