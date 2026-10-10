# Detection and resident inference

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
see [public batch measurements](../docs/detect-batch-2026-09-17.md) and
[postprocessing qualification](../docs/postprocessing-qualification.md).

## Weights

Weights are cached from a pinned revision of
[immich-app/buffalo_l](https://huggingface.co/immich-app/buffalo_l/tree/d09715916a0778919a770c343533641e250b8699),
using Hugging Face Hub. Use `--model det_10g.onnx` or `Scrfd::load(path, options)`
for a local model.

`--offline` or `HF_HUB_OFFLINE=1` requires cached weights. `HF_HOME` and
`HF_HUB_CACHE` select the cache location. `HRX_OFFLINE=1` separately disables
runtime downloads.
