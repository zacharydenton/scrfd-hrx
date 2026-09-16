# CNN activation layout — 2026-09-17

This change optimizes the CNN used by `detect_batch`. The baseline already
contains the unified decoder and batched resize changes; neither changes in
this comparison. All 3x3 convolutions retain their implicit-GEMM WMMA kernels,
FP16 operands and FP32 accumulation.

## CNN changes

Separate physical activation row width from the weight matrix's WMMA tile
width. Previously both were padded to 64-channel boundaries. Activation rows
now use 32-channel boundaries; output stores and fused residual reads guard
the actual row extent. Convolution gathers, pooling and 1x1 projections use
the matching physical stride. No im2col tensor is materialized.

| Real channels | Previous activation width | New width |
| --- | ---: | ---: |
| 28 (stem) | 64 | 32 |
| 80 / 88 | 128 | 96 |
| 224 | 256 | 224 |

The stem already computed only 32 channels but wrote an additional zero-filled
half-row. It now packs weights and activations at 32 channels and removes that
write. Other weight matrices retain their WMMA N padding. The 1x1 projections
also stop multiplying channels removed from their padded inputs. Public dense
head layouts remain unchanged, as do detections and model weights.

## End-to-end detect_batch measurements

Ten warmups, 150 samples/run, before/after/after/before for each batch. Same
frozen HRX source and compiler, gfx1151, 1920x1080 RGB input. Medians below are
medians of the two run medians. No other workloads were stopped. Image decoding,
benchmark input resizing and first-use compilation/preparation are excluded.
[Raw timing samples, fingerprints and convolution profiles](cnn-layout-2026-09-17.json).

| Batch | Before ms/call | After ms/call | Reduction | Tracked resident MiB |
| --- | ---: | ---: | ---: | ---: |
| 1 | 2.543 | 2.499 | 1.7% | 137.7 → 118.5 |
| 4 | 9.071 | 8.571 | 5.5% | 518.7 → 442.2 |
| 16 | 37.226 | 34.799 | 6.5% | 2042.8 → 1736.9 |

The batch-16 resident allocation falls by about 306 MiB (15%). These runtime
counters include prepared input/output buffers, weights and activations with
three inference slots; they are not total process RSS or dedicated VRAM.
The batch-1 gain is small relative to host timing variation. These results do
not imply a corresponding whole-video or faceswap speedup.

## Experiments and validation

Isolated synchronized profiles identified the early 320x320 convolutions and
160x160 stages as substantial costs. Their diagnostic host timings include
individual dispatch/synchronization overhead; sums are not graph latency.
Direct LDS fragment loads, a 128-pixel spatial tile, and 64-wide K staging did
not improve the diagnostic profiles and were rejected. Expanding convolution
K packing to 32-channel boundaries did not establish a public-API improvement
and was also rejected. Only compact activation storage was retained.

All 13 release library tests, including numerical reference, changing batch,
InsightFace fixture, mixed-size public detection, option/cache/replay behavior,
and invalid-input checks pass with unchanged numerical tolerances. Clippy
passes with warnings denied. Before/after detections are byte-identical on the
benchmark input for batches 1, 4 and 16.

```sh
cargo test --release --lib -- --include-ignored --test-threads=1
cargo clippy --all-targets -- -D warnings
cargo run --release --example bench_detect_batch -- 16 150 1920 1080
```
