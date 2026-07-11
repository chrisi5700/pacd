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
  residual region), then **try every primitive type and keep the best** — the one
  that captures the most new interior volume. Heuristic type dispatch (elongated
  region → cylinder, chunky region → box, rounded region → sphere) is a later
  optimisation.

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
size-invariant across parts. Gradients are central finite differences; Adam
drives the update and the quaternion is renormalised each step.

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

Rotations use quaternions — not Euler angles — for well-behaved gradients, and
are renormalised onto the unit sphere after each step. The cylinder is
axis-symmetric, so one component of its quaternion is redundant (harmless gauge
freedom).

---

## Status

The decomposition pipeline is implemented and tested; the renderer + corpus are
the harness around it.

- **Solver (`pacd_solver`):** analytic primitive SDFs (sphere/box/cylinder with
  quaternion orientation), the mesh signed-distance field (point-triangle
  distance + generalized winding number, grid-sampled with trilinear lookup), the
  inscribed inner-loop optimiser (Adam + finite-difference gradients over local
  Monte-Carlo samples), and the greedy driver `decompose(mesh, config)`. Builds
  clean under the strict `llm-vcpkg` preset and is verified end-to-end (synthetic
  cube → one oriented box; a real Thingi10K part → an inscribed primitive set).
- **Viewer (`pacd-viewer`):** decomposes each STL and overlays the fitted
  primitives (solid, colour-coded) on the original mesh (a wireframe cage);
  arrow keys cycle files, `T` toggles the cage, and progress is logged. A **Dear
  ImGui** panel exposes the `SolverConfig` live (budget, vocabulary, resolution,
  sampling, gradient-descent knobs), so you can retune and re-decompose (`R`)
  without restarting. See [`RENDERER.md`](RENDERER.md).
- **Harness:** a `std::expected` STL loader (ASCII + binary), dependency-free
  math, the **100-mesh Thingi10K corpus** (see
  [`resources/README.md`](resources/README.md)), and `render_bridge.hpp`
  (`render::Mesh` → solver `TriMesh`).
- **Next:** BVH acceleration for the mesh SDF (the brute-force grid build
  dominates on large meshes), running decomposition off the UI thread so the
  window stays responsive, and heuristic primitive-type dispatch.

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
