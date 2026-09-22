# Resident composition — 2026-09-23

`Scrfd::submit` now records CNN inference and candidate decoding in the same
cached plan used by the host-canvas API. CPU selection/NMS and final GPU row
gathering remain. Ten GPU tests pass, including changed face/blank inputs,
five option sets, independent CPU decode oracles and warm graph reuse.

Five alternating fresh-process pairs on gfx1151 compare HRX 0.8.4 with this
change and HRX 0.8.5. Batch-one median falls from 7.124 to 5.699 ms (20.0%);
batch-four from 12.162 to 10.928 ms (10.1%). All detection captures are exact.
Each process warms five times and measures 41 requests, including compact CPU
selection and final readback. No concurrent GPU jobs or builds run during timing.
Clocks and desktop activity are not controlled; these are fixture results.

Submissions fall from nine to seven per request. Copy payload falls from
2,566,616 to 1,491,416 bytes at batch one and 10,266,464 to 5,965,664 at batch four.
The benchmark prepares a host oracle, so its peak is not resident-only storage.

Run `cargo run --release --example bench_resident -- 1` (or `4`) with cached
weights. [Shared raw evidence and methodology](https://github.com/zacharydenton/hrx-rs/blob/main/docs/CLIENT-COMPOSITION.md)
retain the discarded first comparison, which accidentally saved identical
executables, and the rebuilt comparison used here.
