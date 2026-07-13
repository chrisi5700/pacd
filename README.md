[![CI](https://github.com/chrisi5700/pacd/actions/workflows/ci.yaml/badge.svg)](https://github.com/chrisi5700/pacd/actions/workflows/ci.yaml)

# pacd — Primitive-Approximate Convex Decomposition

**pacd** approximates triangle-mesh geometry with a small, *heterogeneous* set of
analytic **convex** primitives — spheres, boxes and cylinders — for use as
**physics collision proxies**. Instead of tiling one object with hundreds of
identical boxes, it mixes primitive types so each one earns its place: a cylinder
captures a boss or bore, a sphere a rounded cap, a box the blocky bulk. The payoff
is **parsimony** — far fewer parts for the same fidelity, which is exactly what a
fast collision solver wants.

Every primitive is convex, so their union is a genuine **convex decomposition**.
Keeping the vocabulary convex (no torus) also preserves the clean support-function
/ separating-axis / inertia properties that the solver — and the later
type-selection heuristics — lean on.

The primitives are represented as **signed distance fields (SDFs)** and combined
with a union (`min`), so the whole proxy is one implicit field that CSG operators
compose trivially and the renderer can visualise directly.

---

## Approach

The fitter is built as a **greedy outer loop around a gradient-descent inner
loop**, targeting an **inscribed** proxy (the union of primitives stays *inside*
the mesh surface).

- **Inscribed, not conservative.** A union of primitives can only *grow* the
  covered set, which pairs cleanly with a subset (inside) target and gives a
  monotone, terminable "% of interior filled" stopping rule. A conservative
  (enclosing) target has no such clean stop.

- **Inner loop — grow one primitive (gradient descent).** Optimise a single
  primitive's parameters to fill as much fresh interior as possible while staying
  inscribed. "Stay inside the mesh" is just the SDF **intersection** of the
  primitive with the mesh interior, `max(d_prim, d_mesh)`: the primitive only
  earns credit for the part that survives the intersection, so it inflates like a
  balloon inside the object and is pushed back wherever it pokes out. Volumes are
  softened (`sigmoid(-d/τ)`) so the objective is differentiable, and the union
  uses a **smooth `min`** so gradient flows to all nearby primitives rather than
  only the closest one.

- **Outer loop — place the next primitive (greedy).** Seed the next primitive at
  the **deepest uncovered interior point** (the distance-transform peak of the
  residual region). A PCA of the local interior there yields a **shape frame**
  that warm-starts each candidate's orientation and anisotropic extents (long
  axis along the region's dominant direction) instead of starting axis-aligned,
  so gradient descent *polishes* the fit rather than having to *discover* the
  rotation from scratch. Then **try every primitive type and keep the best** — the
  one that captures the most new interior volume. Fully skipping ill-suited types
  (heuristic dispatch) is a later optimisation.

- **Bounded directional growth.** Elongated features (a bar, a screw shaft, a
  jenga block, a table top) fill best with one large primitive, but an isotropic
  clearance-sized sample window can only reward growth out to a couple of
  clearances — so they used to fragment into short pieces with gaps. Instead the
  window is **marched** out from the seed along the local principal axes: each
  march follows the interior until it hits a wall, a gap, or a **neck** (a point
  where the inscribed clearance necks below a fraction of the seed's), so it hugs
  the feature and never leaps across empty space into a neighbour. The longest
  corridor is taken as the growth axis — a far better length estimate than the
  local PCA variance, which barely sees the length — and the primitive is
  initialised to fill that corridor, so descent *polishes* a full-length fit
  rather than having to *discover* the length within its step budget. The
  neck-stop keeps a constant cross-section from being driven through a bulging
  feature: a chain of spheres stays spheres, not one lumpy cylinder. A seed where
  nothing inscribes cleanly is skipped (its pocket blocked) rather than ending the
  whole run, so one hard spot never strands the remaining budget.

- **Symmetry-aware replication.** The shape's global symmetry is detected up front
  from the SDF — a transform `T` is a symmetry iff the field is invariant under it
  (`d(x) ≈ d(Tx)`), so candidate mirrors and n-fold rotations about the principal
  axes are verified by cheap grid resampling. Each placed primitive is then
  replicated across the detected group, so a symmetric region is filled from a
  single fit. Replicas are validated independently (the same protrusion + fresh
  coverage gates), so an approximate symmetry can never force a protruding or
  redundant primitive — it only ever saves work. On symmetric parts this is 2–4×
  faster and yields a more consistent decomposition.

- **Merge / consolidate.** Greedy placement leaves a crowd of small stitch
  primitives around the few big ones (the diminishing tail). A final pass folds
  adjacent pairs back together: it re-fits **one** primitive to the pair's joint
  interior (warm-started from the region's oriented bounding frame, then the same
  gradient-descent inner loop, so a slightly misaligned pair is polished into a
  proper single fit) and keeps the merge only if that lone primitive re-covers at
  least `merge_retain` of **each member** — counting interior any *surviving*
  primitive still holds, so a merge can never open a hole another primitive already
  fills. The retain test is **per-primitive, not over the union**: a lone fin holds
  only a handful of a big body's nodes, so a union fraction would round its loss to
  zero and let the body swallow it — the cascade that otherwise collapses a whole
  rocket to a single cylinder. Members too small to be a real feature (below
  `MERGE_FEATURE_MIN` grid nodes — aliasing debris the greedy left behind) are
  exempt, so that debris is still absorbed. Applied greedily (largest region first,
  repeated to a fixpoint), it collapses a cluster while leaving distinct features
  standing — on the composite corpus this cuts primitive count *and* nudges mean
  IoU up. Each round's candidate re-fits are independent, so they run in parallel
  across cores, largest-region-first and stopping at the first that sticks; on dense
  meshes (screws) this is a 4–16× speedup over the serial pass, which dominated the
  solver's runtime once the field build was itself parallelised.

- **Swallow redundant leftovers.** A merge re-fits one primitive to a *pair* and
  so leaves them split when one is a redundant left-over sitting mostly inside
  another of a very different shape (a box overlapping a wheel cylinder — the
  joint re-fit can't cover the union). A follow-up pass targets exactly that: a
  primitive already mostly covered by the *rest* is absorbed by **growing its
  dominant coverer over it** (warm-started from that coverer, so it keeps its
  shape and just inflates) and then dropped, provided the grown primitive still
  holds nearly all of both. Being gated on redundancy, it can only ever remove a
  primitive another already covers — a distinct feature (a rocket fin, covered by
  nothing else) is never a candidate, so coverage is preserved while the count
  falls further.

- **Prune redundant left-overs.** A final sweep drops any primitive whose interior
  is *entirely* held by the others — sub-voxel degenerates the greedy placed
  chasing the last fraction of a percent, and anything the merge/swallow left fully
  buried. Removing a primitive that contributes no coverage of its own is exactly
  lossless; done one at a time and re-checked, two primitives that merely overlap
  are never both dropped.

- **Termination.** Stop at *x %* of the interior filled **or** *n* primitives,
  whichever trips first. The tail (concave corners, thin features) is expensive
  and left uncovered by design — acceptable for a collision proxy.

### Fit objective

Each greedy step grows one primitive against a soft-occupancy **volume**
objective (the "balloon"). With soft occupancy `o(d) = σ(−d/τ)`, the primitive is
rewarded for the *fresh* interior it captures — the intersection
`max(d_prim, d_mesh)` of the primitive with the uncovered mesh interior — and
penalised (weight λ) for any volume that protrudes outside the mesh, which
enforces inscription. The integral is estimated by Monte-Carlo sampling **locally
around the seed**, scaled to the local clearance so the objective is
size-invariant across parts. The objective is differentiable in closed form, so
gradients are **analytic** — one pass over the samples rather than a central
finite difference per parameter — and the rotation is updated in its **`so(3)`**
tangent space (`R ← R·exp[θ]`) instead of nudging quaternion components and
renormalising. Adam drives the step, and descent **stops early** once the loss
stops improving (capped at `gd_iterations`).

The mesh SDF itself uses distance-to-nearest-triangle for magnitude and a
generalized winding number for sign — robust to the non-watertight /
non-manifold meshes in the corpus — sampled on a grid and trilinearly
interpolated.

### Primitive parameterisation

| Primitive | Optimised parameters                                  |
|-----------|-------------------------------------------------------|
| Sphere    | centre (3) + radius (1)                               |
| Box       | centre (3) + size (3) + quaternion (4)                |
| Cylinder  | centre (3) + radius (1) + height (1) + quaternion (4) |

Rotations are carried as quaternions — not Euler angles — but are **optimised in
the `so(3)` tangent space** (an incremental body-frame update composed onto the
quaternion), which is better-conditioned than nudging the four quaternion
components and needing to renormalise. They are **initialised from the seed's
local principal axes** (see the outer loop), not the identity, so descent starts
already close to the region's orientation. The cylinder is axis-symmetric, so its
rotation about the long axis is a harmless gauge freedom.

---

## Status

The decomposition pipeline is implemented and tested; the renderer + corpus are
the harness around it.

- **Solver (`pacd_solver`):** analytic primitive SDFs (sphere/box/cylinder with
  quaternion orientation), the mesh signed-distance field (point-triangle
  distance + generalized winding number, grid-sampled with trilinear lookup) --
  built in parallel over a triangle **BVH** that accelerates both the nearest-
  triangle distance (branch-and-bound) and the winding number (Barnes-Hut) to
  ~O(log triangles), ~100x faster than the brute grid build on a 13 k-triangle
  mesh while staying bit-identical -- the
  inscribed inner-loop optimiser (Adam over **analytic** gradients with an
  **`so(3)`** rotation update and convergence early-stop, on local Monte-Carlo
  samples), PCA-warm-started seeding (the seed's local principal axes
  orient and pre-size each candidate), **bounded directional growth** that marches
  the interior corridor so one primitive fills a whole elongated feature (halting
  at walls, gaps and necks), SDF-based symmetry detection with orbit replication
  of each fit, and the greedy driver `decompose(mesh, config)`. Builds clean under
  the strict `llm-vcpkg` preset and is verified end-to-end (synthetic cube → one
  oriented box; an oblique beam → an orientation-aligned fit; an elongated beam →
  one grown full-length primitive; a mirror-symmetric pair → exactly replicated
  fits; a real Thingi10K part → an inscribed primitive set).
- **Viewer (`pacd-viewer`):** decomposes each STL and overlays the fitted
  primitives (solid, colour-coded) on the original mesh (a wireframe cage);
  arrow keys cycle files, `T` toggles the cage, and progress is logged. A **Dear
  ImGui** panel exposes the `SolverConfig` live (budget, vocabulary, resolution,
  sampling, gradient-descent knobs), so you can retune and re-decompose (`R`)
  without restarting. See [`RENDERER.md`](RENDERER.md).
- **Harness:** a `std::expected` STL loader (ASCII + binary), dependency-free
  math, the **100-mesh Thingi10K corpus** (see
  [`resources/README.md`](resources/README.md)), and `render_bridge.hpp`
  (`render::Mesh` → solver `TriMesh`). `tools/score-composites` decomposes a
  directory of composite fixtures and reports per-mesh coverage / spill / IoU
  against the mesh (and the intended part count from the JSON sidecars), so a
  change to the fitter can be judged on real multi-part shapes rather than by eye.
- **Next:** partial/local symmetry (segment first, then detect per-region), and
  running decomposition off the UI thread so the window stays responsive.

See [`RENDERER.md`](RENDERER.md) for the viewer's controls, headless/CI usage and
design notes.

---

## Build & run

Dependencies are resolved through **vcpkg** (set `VCPKG_ROOT` to your checkout
first). Use `dev-vcpkg` for fast iteration and `llm-vcpkg` for the strict gate
(clang-tidy + cppcheck as errors, `-Werror`, ASan/UBSan).

```sh
# Configure + build (fast iteration)
cmake --preset dev-vcpkg
cmake --build --preset dev-vcpkg

# Strict, AI-assisted preset — max safety checks, warnings-as-errors
cmake --preset llm-vcpkg
cmake --build --preset llm-vcpkg

# Run tests / benchmarks
ctest --test-dir build/dev-vcpkg --output-on-failure
./build/dev-vcpkg/bench/bench

# Inspect a part in the viewer
./build/dev-vcpkg/pacd-viewer resources/1036654.stl
```

Configure presets: `dev-vcpkg`, `llm-vcpkg`, `release-vcpkg` (plus
`dev-vcpkg-msvc`).

---

## Project structure

| Path                    | Contents                                                     |
|-------------------------|-------------------------------------------------------------|
| `include/pacd/render/`  | Public renderer headers (`mesh`, `math`, `stl`, `scene`, …) |
| `src/render/`           | Renderer implementation; GL backend confined to `gl/`       |
| `src/viewer_main.cpp`   | `pacd-viewer` CLI front-end                                  |
| `resources/`            | Thingi10K test corpus + selection/download tooling          |
| `tests/`                | Catch2 unit tests                                            |
| `tools/`                | Measurement CLIs (`score-composites` fitter-quality harness) |
| `bench/`                | Google Benchmark microbenchmarks                             |
| `playground/`           | Throwaway prototyping                                        |
| `cmake/`                | Target helpers, compiler settings, tooling                  |

New targets are declared in the top-level `CMakeLists.txt` with the
`target_add_library` / `target_add_executable` helpers, which apply the
configured warnings, sanitizers and analysis automatically. Implementation goes
under `src/`, public headers under `include/`.

---

## Dependencies

`fmt`, `spdlog` (logging), `glfw3` + `glad` (GL 3.3 loader) + `imgui` (config
panel, viewer only), system OpenGL, `catch2` (tests), `benchmark` (benchmarks) —
all via `vcpkg.json`.
