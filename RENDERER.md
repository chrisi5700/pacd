# pacd renderer

An interactive OpenGL mesh viewer. Given STL files it decomposes each into convex
primitives (via `pacd_solver`) and shows them as solid, colour-coded shapes
overlaid on the original mesh (drawn as a wireframe cage), so you can compare the
approximation against the source. A **Dear ImGui** panel exposes the solver
config live, so you can retune and re-decompose without restarting. Library lives
in `include/pacd/render/` + `src/render/`; the CLI front-end is
`src/viewer_main.cpp` (target `pacd-viewer`).

## Build & run

```bash
# Use a fast build for interactive use: the llm preset's sanitizers + coverage
# make decomposition ~10-50x slower.
cmake --preset release-vcpkg
cmake --build --preset release-vcpkg --target pacd-viewer

# decompose a part and view the overlay (original cage + fitted primitives)
./build/release-vcpkg/pacd-viewer resources/69264.stl

# pass several files -> Left/Right arrows cycle through them (each is decomposed
# on demand; progress is logged to stderr)
./build/release-vcpkg/pacd-viewer resources/*.stl

# no argument -> built-in primitive gallery (box/sphere/cylinder/cone/plane)
./build/release-vcpkg/pacd-viewer
```

The whole thing builds clean under the strict `llm-vcpkg` preset
(clang-tidy + cppcheck as errors, `-Werror`, ASan/UBSan). When running the
sanitized build, set `ASAN_OPTIONS=detect_leaks=0` to silence GPU-driver leak
reports.

## Controls

| Input                 | Action                             |
|-----------------------|------------------------------------|
| Left-drag             | orbit around the target            |
| Middle / right-drag   | pan                                |
| Scroll                | zoom (dolly)                       |
| `Left` / `Right`      | cycle through the loaded meshes    |
| `R`                   | re-decompose with the current knobs|
| `T`                   | toggle the original mesh (cage)    |
| `F`                   | frame / fit the scene              |
| `W`                   | toggle wireframe (all objects)     |
| `S`                   | save `pacd-shot-N.png`             |
| `Esc`                 | quit                               |

Input over the ImGui panel is routed to the panel (it won't orbit the camera or
fire shortcuts). The **config panel** edits the live `SolverConfig` — primitive
budget/coverage, the primitive vocabulary (sphere/box/cylinder), SDF resolution,
sample count, GD iterations and learning rate, plus the softness/penalty knobs
under *Advanced* — then **Re-decompose** (or `R`) re-runs the fit on the current
mesh. Decomposition is synchronous, so the window freezes while it runs; progress
streams to stderr.

## Headless / CI

`--hidden --frames N --screenshot out.png` renders offscreen and writes a PNG,
so renders can be produced and diffed without a display:

```bash
./build/llm-vcpkg/pacd-viewer --hidden --frames 3 \
    --screenshot /tmp/part.png resources/1036654.stl
```

Other flags: `--width/--height`, `--wire` (start in wireframe), `--flat` (keep
raw facet normals instead of smoothing).

## Design notes

- **Software-independent core, OpenGL backend.** `Mesh`, `Aabb`, `Material`,
  `DirectionalLight`, `Scene`, `OrbitCamera` and the STL/primitive builders are
  GL-free and unit-testable. GLSL/GLFW/glad are confined to the `.cpp` files of
  `shader`, `gpu_mesh` and `viewer`, and are not exposed in any public header
  (the window handle is a type-erased `void*`). A different backend could be
  dropped in behind the same `Scene` API.
- **Shading:** two-sided Blinn-Phong with a two-light key/fill studio setup, so
  open, non-manifold and inside-out CAD meshes still shade correctly.
- **Crease-aware normals (default for STL).** `with_crease_normals()` averages
  face normals across a shared edge only below a crease angle (35 deg), so flat
  faces and sharp edges stay crisp while curved bores/fillets stay smooth.
  Unconditional smoothing "melts" hard-surface CAD parts (domed faces); `--flat`
  keeps pure per-facet normals to inspect the raw tessellation.
- **Sanitizer runs are clean.** The GPU driver leaks internal buffers at exit;
  the viewer registers an `__lsan_default_suppressions` hook so the ASan/UBSan
  build reports nothing spurious — no `detect_leaks=0` needed.
- **STL loader** auto-detects ASCII vs binary by the size formula (not the
  `solid` prefix, which binary files also use) and returns `std::expected`.
- **Convex-decomposition comparison:** load the source mesh plus each hull as
  separate `Object`s; they are overlaid at the origin, each in a palette colour.
  `load_stl_scene()` (in `scenes.hpp`) does this for a list of paths. Toggle
  wireframe (`W`) to see hull coverage against the original.
- **Math** is a small dependency-free `Vec3/Vec4/Mat4` (column-major, so it
  feeds GL uniforms directly).

## Dependencies

`glfw3` and `glad` (GL 3.3 core loader) and `imgui` (with its `glfw-binding` +
`opengl3-binding` features, for the config panel), added to `vcpkg.json`; system
OpenGL via CMake `find_package(OpenGL)`.

`Viewer` owns the ImGui lifecycle (context, GLFW/GL3 backends, per-frame
new-frame/render) but stays UI-agnostic: the app hands it a `set_on_gui` callback
that issues the widget calls, so the render library never learns about
`SolverConfig`. Because the callback runs mid-frame, it only *records* requests
(re-decompose / cycle); the app drains them via `while (viewer.pump()) { … }` —
outside the ImGui frame, where the blocking decompose is safe.
