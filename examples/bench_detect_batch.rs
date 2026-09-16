//! Warm public detect_batch latency, including preprocessing and host results.
use anyhow::{Result, ensure};
use scrfd_hrx::{DetectionOptions, Image, Options, Scrfd};
use std::time::Instant;
fn main() -> Result<()> {
    let args = std::env::args().collect::<Vec<_>>();
    let batch = args
        .get(1)
        .map(|s| s.parse())
        .transpose()?
        .unwrap_or(1usize);
    let samples = args
        .get(2)
        .map(|s| s.parse())
        .transpose()?
        .unwrap_or(100usize);
    let width = args
        .get(3)
        .map(|s| s.parse())
        .transpose()?
        .unwrap_or(1920u32);
    let height = args
        .get(4)
        .map(|s| s.parse())
        .transpose()?
        .unwrap_or(1080u32);
    ensure!(
        (1..=64).contains(&batch) && samples >= 10,
        "invalid batch/samples"
    );
    let image = image::load_from_memory(include_bytes!("../tests/fixtures/t1.png"))?.to_rgb8();
    let image =
        image::imageops::resize(&image, width, height, image::imageops::FilterType::Triangle);
    let inputs = vec![
        Image {
            rgb: image.as_raw(),
            width: width as usize,
            height: height as usize
        };
        batch
    ];
    let model = Scrfd::from_pretrained(Options {
        max_batch: batch,
        ..Default::default()
    })?;
    let run = || model.detect_batch(&inputs, DetectionOptions::default());
    for _ in 0..10 {
        std::hint::black_box(run()?);
    }
    let before = model.context().runtime().statistics();
    let mut times = Vec::new();
    for _ in 0..samples {
        let start = Instant::now();
        std::hint::black_box(run()?);
        times.push(start.elapsed().as_secs_f64() * 1000.);
    }
    let after = model.context().runtime().statistics();
    let output = run()?;
    if let Some(path) = args.get(5) {
        std::fs::write(path, serde_json::to_vec(&output)?)?;
    }
    times.sort_by(f64::total_cmp);
    println!(
        "{}",
        serde_json::json!({"batch":batch,"width":width,"height":height,"samples":samples,
        "resident_bytes":after.live_bytes,"peak_bytes":after.peak_bytes,
        "median_ms":times[samples/2],"p95_ms":times[(samples*95).div_ceil(100)-1],
        "downloaded_bytes_per_call":(after.downloaded_bytes-before.downloaded_bytes)/samples as u64,
        "device_copied_bytes_per_call":(after.device_copied_bytes-before.device_copied_bytes)/samples as u64,
        "submissions_per_call":(after.submissions-before.submissions)/samples as u64,"warm_ms":times})
    );
    Ok(())
}
