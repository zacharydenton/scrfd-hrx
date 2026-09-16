# Public detect_batch unification — 2026-09-17

`detect`, `detect_batch`, `detect_letterboxed` and the resident APIs now share
one GPU candidate decoder and one stable CPU selector. CPU decoding is compiled
only for tests. The prior resident-only speedup did not apply to `detect_batch`;
these measurements call that public API from RGB bytes through returned faces.

The host API records letterboxing, CNN inference and parallel compaction in a
single graph per chunk. It maps coherent status and score/box summaries for
selection, then reads only the selected full rows. It neither copies/scans
dense CNN heads on CPU nor gathers rows back onto the GPU just to read them.
Resident callers use the same decoder/selector and gather selected device rows.

Consecutive equal-sized images are resized as a batch. Individual input slices
map to disjoint views of one host-visible allocation, so the common equal-size
case needs no canvas concatenation copy. Mixed-size images remain supported
and are concatenated by run. No image padding, arithmetic or filtering changes.

## Warm public API latency

1920x1080 input frames made from the existing face fixture, default detection
options, max_batch equal to the measured batch. Ten warmups and 150 samples per
run, before/after/after/before order; values are medians of two run medians.
Both binaries use identical frozen HRX 0.7.0 working sources. No other running
workload was stopped. [Raw samples and fingerprints](detect-batch-2026-09-17.json).

| Batch | Before ms/call | After ms/call | Reduction |
| --- | ---: | ---: | ---: |
| 1 | 2.582 | 2.540 | 1.6% |
| 4 | 9.397 | 9.064 | 3.5% |
| 16 | 39.266 | 37.311 | 5.0% |

The single-frame change is small relative to run variation. Batch 16 removes
19,660,800 bytes of device canvas copies per call; batch 4 removes 4,915,200.
The old API already used coherent heads (zero GPU readback-copy counter), but
copied 1,075,200 bytes of dense head data per image into CPU-owned vectors and
scanned every anchor. The new API maps compact outputs without that CPU copy.
Both versions use one GPU submission per chunk. These are detector timings,
not whole faceswap/video throughput. First-use model and graph preparation,
image decoding and benchmark input resizing are excluded.

## Correctness and reproduction

All 13 GPU/library tests pass and Clippy passes with warnings denied. The new
public-API test compares against independently CPU-decoded heads for changing
thresholds, capacities, NMS settings and ranking; checks overflow/recovery,
empty input, partial chunks, replay and mixed-size grouping; and asserts no
warm copies or allocations for equal-sized batches. The resident decoder's
existing exact numerical, nonfinite and stable-order checks remain in place.
Benchmark detections are byte-identical before/after for batches 1, 4 and 16.

```sh
cargo test --release --lib -- --include-ignored --test-threads=1
cargo clippy --all-targets -- -D warnings
cargo run --release --example bench_detect_batch -- 16 150 1920 1080
```
