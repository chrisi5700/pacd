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

### Fit metric

Fit is measured as an **SDF-field distance**: the mesh's signed distance field
vs. the proxy field `min_i d_i` over sampled points, narrow-band weighted around
the surface, and made **asymmetric** to encode the inscribed preference. The mesh
SDF uses distance-to-nearest-triangle for magnitude and a generalized winding
number for sign (robust to the non-watertight / non-manifold meshes in the test
corpus).

### Primitive parameterisation

| Primitive | Parameters                                             | DoF |
|-----------|--------------------------------------------------------|-----|
| Sphere    | centre + radius                                        | 4   |
| Box       | centre + rotation + half-extents (+ optional rounding) | 9   |
| Cylinder  | centre + axis (2 DoF, axis-symmetric) + radius + half-height | 7 |

Rotations use quaternions / exponential-map (not Euler angles) for well-behaved
gradients.

---

## Status

Greenfield on the algorithm; the harness around it is in place.

- **Built:** the interactive OpenGL mesh viewer (`pacd-viewer`) for inspecting
  parts and overlaying a source mesh against its approximation, a `std::expected`
  STL loader (ASCII + binary), dependency-free `Vec3/Mat4` math, crease-aware
  normals, and a curated **100-mesh Thingi10K test corpus** (see
  [`resources/README.md`](resources/README.md)).
- **Planned:** the mesh-SDF builder, the analytic primitive-SDF library
  (sphere/box/cylinder + smooth CSG), the inscribed inner-loop optimiser, and the
  greedy driver.

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

`fmt`, `spdlog` (logging), `glfw3` + `glad` (GL 3.3 loader, viewer only), system
OpenGL, `catch2` (tests), `benchmark` (benchmarks) — all via `vcpkg.json`.
