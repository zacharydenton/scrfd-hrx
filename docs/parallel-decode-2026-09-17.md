# Parallel resident decoding — 2026-09-17

Historical measurements before [unifying all detection entry points](detect-batch-2026-09-17.md).
`detect_batch` now uses the same GPU decoder; the CPU decoder is a test oracle only.

The resident SCRFD path now classifies 16,800 anchors in parallel, scans 66
block totals, and decodes accepted anchors directly into compact rows and
score/box summaries. Stable anchor order, exact sigmoid values, CPU NMS and
landmark gathering are preserved. This removes the single-lane anchor loop,
capacity-wide candidate clear and separate summary compaction pass.

## Measurements

Same frozen HRX 0.7.0 working tree for both variants; before/after/after/before
on gfx1151. Medians below combine six warm samples per case. These diagnostic
samples are few and other GPU workloads were left running.
[Raw results and runtime fingerprint](parallel-decode-2026-09-17.json).

| Resident postprocessing case | Before ms | After ms |
| --- | ---: | ---: |
| 256 disjoint | 5.638 | 0.621 |
| 256 overlapping | 5.397 | 0.441 |
| 256 ranked | 5.512 | 0.581 |
| 4,096 disjoint | 35.462 | 18.632 |
| 4,096 overlapping | 17.689 | 0.571 |
| 4,096 ranked | 34.555 | 18.414 |

Scope includes decode, metadata readback, CPU NMS/ranking, selected-index
upload, GPU gather and final selected-row readback. It excludes CNN execution
and initial head upload. The host `detect_batch` path still uses CPU decoding;
these numbers must not be interpreted as whole-detector speedups. Dense
non-overlapping cases remain limited by CPU NMS.

Scratch increases by 68,376 bytes per image per prepared slot (three slots).
Only accepted candidate rows are written; their padding is explicitly zeroed.
Readback remains 8 status bytes/image, 20 bytes/candidate and 64 bytes/selected
row when a host result is requested. Every scratch slot is overwritten before
use. Overflow and nonfinite values still produce errors.

## Validation

All 12 library tests passed, including full model reference/replay, the
InsightFace fixture, exact CPU decoding parity, capacity/threshold/nonfinite
cases, and stable ordering across anchor-block and pyramid-level boundaries.
The new three-image boundary test includes an empty middle image, final anchor,
exact-capacity and overflow cases. Clippy passes with warnings denied.

```sh
cargo test --release --lib -- --include-ignored --test-threads=1
cargo clippy --all-targets -- -D warnings
```
