# Spatter input decks

The small JSON decks in `standard-suite/` are byte-for-byte copies of
[`hpcgarage/spatter`](https://github.com/hpcgarage/spatter) commit
`ec8923711f8dc21eedff7189f12b02eb06845d2f`. Their SHA-256 values are:

| Input | SHA-256 |
| --- | --- |
| `standard-suite/basic-tests/gpu-stream.json` | `34d10c083f159800fbfd05fa50ae8041a871e3e9872cf8692c9d8c0523ebb4c9` |
| `standard-suite/app-traces/lulesh.json` | `9073035ecf77e7fde65262f782286207e76cca24312b2e01688b038901d021ee` |
| `standard-suite/app-traces/amg_gpu.json` | `1b50a9a6dbc612c718b2654462c9dda04674b61b5f070651a327b482c4e1f027` |

The 517,377,170-byte LANL xRAGE input `spatter.json` is preserved locally
under `../../evaluation/dependency-snapshot/input/` through the
artifact inventory, with SHA-256
`7325525ada0dacb6e1206717d242f6721b6d8da77718506fd909444c388f7733`.
The large input snapshot is ignored by Git. Its original copy came from the
LANL Spatter trace data used for the xRAGE asteroid benchmark.
