//! Warm detection latency, measured where the model is rather than through a
//! caller's queues.
//!
//! Measures `detect_letterboxed` alone, one call at a time, without application
//! queues or inference-slot contention. This is not end-to-end application
//! latency and does not include image decoding or letterboxing.
//!
//! Input is the InsightFace fixture with nearest-neighbor resizing, top-left
//! alignment and zero padding. Face count matters: suppression
//! and gathering scale with candidates, so a synthetic canvas with no faces
//! measures a different code path than a real one.
//!
//! Compare two revisions with criterion's own baselines:
//!
//!     cargo bench --bench detect -- --save-baseline before
//!     git checkout <other>
//!     cargo bench --bench detect -- --baseline before

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use scrfd_hrx::{DetectionOptions, Options, SIZE, Scrfd};
use std::hint::black_box;

/// Top-left letterbox into a 640 square, returning the canvas and its scale.
fn letterbox(rgb: &[u8], width: usize, height: usize) -> (Vec<u8>, f32) {
    let scale = (SIZE as f32 / width as f32).min(SIZE as f32 / height as f32);
    let (w, h) = (
        ((width as f32 * scale) as usize).min(SIZE),
        ((height as f32 * scale) as usize).min(SIZE),
    );

    let mut canvas = vec![0u8; SIZE * SIZE * 3];
    for y in 0..h {
        let source_y = ((y as f32 / scale) as usize).min(height - 1);
        for x in 0..w {
            let source_x = ((x as f32 / scale) as usize).min(width - 1);
            let from = (source_y * width + source_x) * 3;
            let to = (y * SIZE + x) * 3;
            canvas[to..to + 3].copy_from_slice(&rgb[from..from + 3]);
        }
    }
    (canvas, scale)
}

fn detect(c: &mut Criterion) {
    let path = match std::env::var_os("SCRFD_MODEL") {
        Some(path) => std::path::PathBuf::from(path),
        None => scrfd_hrx::hub::weights(false).expect("weights"),
    };

    let image = image::load_from_memory(include_bytes!("../tests/fixtures/t1.png"))
        .expect("fixture")
        .to_rgb8();
    let (width, height) = image.dimensions();
    let (canvas, scale) = letterbox(&image.into_raw(), width as usize, height as usize);
    let shape = [width as usize, height as usize];

    let mut group = c.benchmark_group("detect_letterboxed");

    // 16 is the analysis pass's detect batch; 1 shows what the batch is worth.
    for batch in [1usize, 16] {
        // `&mut` before the shared context, `&self` after; this binds for both.
        #[allow(unused_mut)]
        let mut model = Scrfd::load(
            &path,
            Options {
                device: 0,
                max_batch: batch,
            },
        )
        .expect("load");

        let canvases: Vec<u8> = canvas.repeat(batch);
        let scales = vec![scale; batch];
        let shapes = vec![shape; batch];

        // One call is `batch` images, so report per-image throughput.
        group.throughput(criterion::Throughput::Elements(batch as u64));
        group.bench_with_input(BenchmarkId::from_parameter(batch), &batch, |b, _| {
            b.iter(|| {
                let found = model
                    .detect_letterboxed(
                        black_box(&canvases),
                        &scales,
                        &shapes,
                        DetectionOptions::default(),
                    )
                    .expect("detect");
                black_box(found)
            })
        });
    }

    group.finish();
}

criterion_group!(benches, detect);
criterion_main!(benches);
