# Spatter input decks

The 13 JSON decks in `standard-suite/` are byte-for-byte copies of
[`hpcgarage/spatter`](https://github.com/hpcgarage/spatter) commit
`ec8923711f8dc21eedff7189f12b02eb06845d2f`. The
[manifest](standard-suite/manifest.csv) records each file's SHA-256 and byte
size. The upstream `LICENSE` and `COPYING` are included beside the decks.
The generated [native-size inventory](../evaluation/native-workload-footprint.csv)
enumerates their 114 cases; it does not imply every case has run on Radiance.

The 517,377,170-byte LANL xRAGE input `spatter.json` is preserved locally
under `../../evaluation/dependency-snapshot/input/` through the
artifact inventory, with SHA-256
`7325525ada0dacb6e1206717d242f6721b6d8da77718506fd909444c388f7733`.
The large input snapshot is ignored by Git. Its original copy came from the
LANL Spatter trace data used for the xRAGE asteroid benchmark.
