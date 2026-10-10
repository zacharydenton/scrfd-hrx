# scrfd-hrx

SCRFD face detection in Rust, using [Loom](https://github.com/ROCm/hrx-system)
kernels and [HRX](https://github.com/zacharydenton/hrx-rs) on AMD Strix Halo.
Runs InsightFace's `det_10g` model and returns boxes, scores and five landmarks.

Warm detection measured **2.54 ms per 1080p image** and **9.06 ms per batch of four**
on `gfx1151`. The September 2026 measurements cover RGB input through returned
faces, excluding loading, graph preparation and image decoding.
[Benchmark details](docs/detect-batch-2026-09-17.md).

## Build and run

Requires Rust 1.91+, Linux x86-64, glibc 2.43+ and Radeon 8060S (`gfx1151`).
From the checkout:

```sh
cargo run --release -- --input photo.jpg --output faces.json
```

Reads PNG or JPEG and writes JSON. Omit `--output` for stdout.
The pinned model and HRX runtime download on first use. Use `--model det_10g.onnx`
for local weights or `--offline` for cached weights. `--benchmark 100` measures
warm detection; `--max-batch 4` reduces resident batch capacity.

## Rust API

```rust
use scrfd_hrx::{DetectionOptions, Image, Options, Scrfd};

fn main() -> anyhow::Result<()> {
    let image = image::open("photo.jpg")?.to_rgb8();
    let mut model = Scrfd::from_pretrained(Options::default())?;
    let faces = model.detect(
        Image {
            rgb: image.as_raw(),
            width: image.width() as usize,
            height: image.height() as usize,
        },
        DetectionOptions::default(),
    )?;
    println!("{faces:#?}");
    Ok(())
}
```

Images are resized and letterboxed to 640×640; results use original-image
coordinates. `detect_batch` accepts multiple images. `DetectionOptions` controls
score thresholds, NMS and result limits. Resident outputs can feed
[ArcFace](https://github.com/zacharydenton/arcface-hrx) in the same HRX context.
[Batching, postprocessing and device API](docs/inference.md).

## Development

```sh
cargo test
cargo clippy --all-targets -- -D warnings
cargo test --release -- --include-ignored --test-threads=1
```

Ignored tests require model weights and, for inference, `gfx1151`.
`SCRFD_MODEL` selects local weights. [Numerical checks](docs/postprocessing-qualification.md) ·
[Batch latency and memory](docs/cnn-layout-2026-09-17.md) · [Changelog](CHANGELOG.md)

## License

[Apache-2.0](LICENSE). Model weights have separate terms;
see [third-party notices](THIRD_PARTY_NOTICES.md).
