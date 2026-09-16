# Postprocessing qualification

The hybrid pipeline uses GPU decode/compaction, CPU stable NMS/ranking, and
shared HRX GPU row gathering. CPU selection receives only scores and boxes;
landmarks stay on-device. This is the production path, not a failure fallback.
The former GPU NMS kernel has been removed.

Run the model-free dense qualification on a provisioned gfx1151 machine:

```sh
cargo test --release dense_decode_qualification -- --ignored --nocapture --test-threads=1
```

The fixture covers 256 and 4,096 candidates, repeated scores, disjoint boxes,
heavy overlap, and top-16 center ranking. It checks exact score, box, landmark
and ordering parity against the frozen CPU oracle; compact readback volume;
and eventual warm storage release with stable graph/stream counts. Additional
tests cover threshold boundaries, small candidate capacities, empty images,
invalid predictions and overflow.

Hybrid timings include GPU decoding, status and candidate metadata readback,
CPU selection, selected-index upload, GPU gathering and selected-row readback.
Dense head upload is excluded for both the hybrid and former GPU measurements.
CPU-only timing starts from already-host-resident heads. Preparation includes
compilation/loading when needed. This is not a full detector benchmark.

Observed warm results on gfx1151 (2026-09-16, release build):

| 4,096-candidate case | Initial serial GPU | Former wave GPU | Hybrid | CPU-only oracle |
| --- | ---: | ---: | ---: | ---: |
| Disjoint | 4,675 ms | 144 ms | 34 ms | 12.4 ms |
| Heavy overlap | 3,330 ms | 83 ms | 16 ms | 0.18 ms |
| Top-16 ranked | 17,258 ms | 323 ms | 35 ms | 12.7 ms |

The hybrid improves these local resident-head cases by approximately 4–9x over
the wave implementation. It does not beat the host-only oracle: GPU decoding
and transfers still have a cost. These are diagnostic measurements, not portable
latency guarantees. Typical 256-candidate hybrid cases take about 5 ms.

Per submission, CPU selection downloads 8 status bytes per image and 20 bytes
per accepted candidate. Selected indices are uploaded in a power-of-two bucket
(fewer than twice the selected count); gathering returns only the actual selected
rows. A host result additionally reads 64 bytes per selected row. Empty results
do not create a gather plan. Three private slots per cached gather shape retain
output ownership, and the shared context budget covers allocations.

The model reference/replay and InsightFace fixtures pass. The downstream
resident detector-to-embedding integration also passes exact box, landmark and
embedding parity, including empty images and bounded transfer accounting.
