#pragma once

#include <expected>
#include <span>
#include <string>
#include <vector>

#include "pacd/render/math.hpp"
#include "pacd/render/scene.hpp"

namespace pacd::render {

// A set of visually distinct object colours, cycled when a scene has more parts
// than entries (e.g. the pieces of a convex decomposition).
[[nodiscard]] const std::vector<Vec3>& palette();

// A gallery of the built-in primitives resting on a ground plane, for smoke
// tests and to demonstrate Phong shading.
[[nodiscard]] Scene make_primitive_gallery();

// Load one or more STL meshes into a single scene, each given a palette colour
// and overlaid at the origin. Overlaying is what the convex-decomposition
// comparison needs: the original mesh plus its hull pieces in one view.
// With `weld` the meshes are smooth-shaded; otherwise raw facet normals are
// kept (useful to inspect tessellation).
[[nodiscard]] std::expected<Scene, std::string> load_stl_scene(std::span<const std::string> paths,
                                                              bool weld = true);

}  // namespace pacd::render
