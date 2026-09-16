# scrfd-hrx

SCRFD face detection in Rust for AMD GPUs, powered by
[HRX](https://github.com/zacharydenton/hrx-rs). Library and CLI for InsightFace's
`det_10g` model, returning bounding boxes, scores and five facial landmarks.

Requires Rust 1.91+, Linux x86-64, glibc 2.43+ and a Radeon 8060S (`gfx1151`).
The runtime and pretrained weights download automatically on first use.

## CLI

```bash
cargo run --release -- --input photo.jpg --output faces.json
```

Reads PNG or JPEG and writes JSON. Omit `--output` to print to stdout.
Use `--benchmark 100` for warm timings or `--max-batch 4` to reduce memory use.

## Rust

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

Input is packed uint8 **RGB**, resized and letterboxed to 640×640. Results use
original-image coordinates. For face embeddings, pass the RGB image and
landmarks to [arcface-hrx](https://github.com/zacharydenton/arcface-hrx).

`detect_batch` handles multiple images; `detect_letterboxed` accepts prepared
640×640 RGB canvases. `max_batch` defaults to 16 (range 1–64).
`DetectionOptions` controls score filtering, NMS and result limits.

Decoding and candidate compaction run on the GPU. Stable NMS and result ranking
run on the CPU using only candidate scores and boxes (20 bytes per candidate).
All detection entry points share this decoder and selector. `detect_batch`
composes resizing, CNN inference and decoding into one graph per chunk. It
reads compact metadata and selected rows directly from coherent host-visible
storage, without copying dense CNN heads. Equal-sized consecutive images share
batched resize storage, eliminating the canvas concatenation copy. The resident
API gathers the selected rows on the GPU for downstream model consumption.
Exceeding `max_candidates` is an error, not silent truncation; non-finite model
predictions are also rejected.

For shared-context pipelines, `submit` accepts resident letterboxed canvases
and returns `postprocess::Detections`. Submission waits for candidate metadata
and CPU selection, then enqueues asynchronous gathering. Its `rows` are packed
F32 `[total_faces,16]`; `counts` waits for gathering and returns per-image counts;
`landmarks` returns checked resident views directly usable by ArcFace. `wait`
downloads selected rows. Both `detect_batch` and resident `letterbox` use shared GPU integer
bilinear resize, byte-exact against the former CPU implementation on the tested
fixtures. The hybrid path preserves stable score ties and greedy NMS ordering;
see [public batch measurements](docs/detect-batch-2026-09-17.md) and
[postprocessing qualification](docs/postprocessing-qualification.md).

## Weights

Weights are cached from a pinned revision of
[immich-app/buffalo_l](https://huggingface.co/immich-app/buffalo_l/tree/d09715916a0778919a770c343533641e250b8699),
using Hugging Face Hub. Use `--model det_10g.onnx` or `Scrfd::load(path, options)`
for a local model.

`--offline` or `HF_HUB_OFFLINE=1` requires cached weights. `HF_HOME` and
`HF_HUB_CACHE` select the cache location. `HRX_OFFLINE=1` separately disables
runtime downloads.

## Development

```bash
cargo test
cargo clippy --all-targets -- -D warnings
cargo test --release -- --include-ignored --test-threads=1
```

Ignored tests require model weights and, for inference, `gfx1151`. Set
`SCRFD_MODEL` to use local weights. Run `cargo doc --open` for API docs.

See [CNN batch latency and memory measurements](docs/cnn-layout-2026-09-17.md),
[earlier kernel measurements](docs/kernels-2026-09-10.md) and the
[changelog](CHANGELOG.md).

## License

[Apache-2.0](LICENSE). Model weights have separate terms; see
[third-party notices](THIRD_PARTY_NOTICES.md).
