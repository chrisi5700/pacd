# Convex-decomposition test corpus

100 triangle meshes (`*.stl`) selected from the **Thingi10K** dataset, curated to
exercise a convex-decomposition pipeline across a wide range of topologies. All
files are plain STL (a mix of ASCII and binary) so any loader — assimp, a small
custom parser, `trimesh`, etc. — can read them without a B-rep/STEP tessellation
step.

- **Source:** [Thingi10K](https://github.com/Thingi10K/Thingi10K)
  (`raw_meshes/{file_id}.stl` on the [HF mirror](https://huggingface.co/datasets/Thingi10K/Thingi10K)).
- **Count:** 100 meshes, ~158 MB total, 694–174 172 triangles each.
- **Encoding:** 47 ASCII STL, 53 binary STL (deliberate — tests both code paths).
- **Licensing:** per-file, see the `license` column in `manifest.csv`
  (Thingi10K aggregates Creative Commons / public-domain Thingiverse uploads).

## Layout

```
resources/
├── *.stl            # 100 meshes, named {thingi_file_id}.stl
├── manifest.csv     # one row per mesh: topology metrics + provenance
└── tools/           # scripts to reproduce / extend the selection
    ├── select.py    # deterministic topology-bucketed picker
    └── download.py  # fetches selected STLs, writes manifest.csv
```

## Topology buckets

Selection is driven by topology (not filename), then biased toward mechanical /
CAD-like parts via Thingiverse tags — every mesh matched at least one mechanical
keyword (bracket, gear, bolt, flange, mount, bearing, …). Buckets:

| bucket           | n  | what it stresses                                        |
|------------------|----|---------------------------------------------------------|
| `genus0_solid`   | 14 | simple closed manifold solids (baseline)                |
| `genus1`         | 12 | one through-hole / handle                               |
| `genus2`         |  8 | two handles                                             |
| `genus3_5`       | 10 | few handles                                             |
| `genus6_11`      |  8 | many handles                                            |
| `genus12plus`    |  8 | high genus / lattice-like                               |
| `multi_2_3`      |  8 | 2–3 disconnected components                             |
| `multi_4_9`      |  6 | 4–9 components                                          |
| `multi_10plus`   |  4 | ≥10 components (loose assemblies)                       |
| `open_surface`   |  8 | non-watertight (has boundary edges) — robustness        |
| `nonmanifold`    |  8 | vertex/edge non-manifold — robustness                   |
| `selfintersect`  |  6 | closed & manifold but self-intersecting — robustness    |

Genus is computed as `components − euler_characteristic/2` and is only reported
for closed, manifold, oriented meshes (blank in `manifest.csv` otherwise). The
last three buckets are intentionally "dirty" input so you can verify the pipeline
degrades gracefully (repair, skip, or warn) rather than crashing.

## `manifest.csv` columns

`file_id, bucket, genus, comps, faces, verts, closed, vertex_manifold,
edge_manifold, oriented, self_intersections, boundary_edges, solid, euler,
stl_kind, stl_tris, bytes, category, subcategory, name, tags, license, thing_id`

`stl_tris` was parsed back out of each downloaded file and cross-checks exactly
against the dataset's `faces` count (0 mismatches), so the meshes are intact.

## Reproduce or extend

The scripts read four small metadata CSVs from the Thingi10K HF mirror
(`geometry_data.csv`, `input_summary.csv`, `contextual_data.csv`, `tag_data.csv`).
Fetch them into `.thingi-meta/`, then:

```bash
python3 resources/tools/select.py     # edit the `buckets` table to retarget
python3 resources/tools/download.py   # pulls STLs into resources/, writes manifest.csv
```

To scale up (e.g. add pure-CAD parts with fillets/threads), the same pattern
works against the [ABC dataset](https://deep-geometry.github.io/abc-dataset/)
STL chunks — at the cost of multi-GB downloads.
