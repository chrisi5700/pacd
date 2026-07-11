# pacd renderer

An interactive OpenGL mesh viewer for inspecting parts and comparing a source
mesh with its convex decomposition. Library lives in `include/pacd/render/` +
`src/render/`; the CLI front-end is `src/viewer_main.cpp` (target
`pacd-viewer`).

## Build & run

```bash
cmake --preset llm-vcpkg
cmake --build --preset llm-vcpkg --target pacd-viewer

# interactive: inspect a part
./build/llm-vcpkg/pacd-viewer resources/1036654.stl

# no argument -> built-in primitive gallery (box/sphere/cylinder/cone/plane)
./build/llm-vcpkg/pacd-viewer

# overlay several meshes in distinct colours (source + hull pieces)
./build/llm-vcpkg/pacd-viewer part.stl hull_00.stl hull_01.stl ...
```

The whole thing builds clean under the strict `llm-vcpkg` preset
(clang-tidy + cppcheck as errors, `-Werror`, ASan/UBSan). When running the
sanitized build, set `ASAN_OPTIONS=detect_leaks=0` to silence GPU-driver leak
reports.

## Controls

| Input                 | Action                    |
|-----------------------|---------------------------|
| Left-drag             | orbit around the target   |
| Middle / right-drag   | pan                       |
| Scroll                | zoom (dolly)              |
| `F`                   | frame / fit the scene     |
| `W`                   | toggle wireframe          |
| `S`                   | save `pacd-shot-N.png`    |
| `Esc`                 | quit                      |

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

`glfw3` and `glad` (GL 3.3 core loader), added to `vcpkg.json`; system OpenGL
via CMake `find_package(OpenGL)`.
