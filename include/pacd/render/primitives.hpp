#pragma once

#include "pacd/render/math.hpp"
#include "pacd/render/mesh.hpp"

namespace pacd::render {

// Analytic primitive meshes, centred on the origin, with correct normals ready
// for Phong shading. Segment/ring counts trade smoothness for triangle count.

[[nodiscard]] Mesh make_box(Vec3 size = {1.0F, 1.0F, 1.0F});
[[nodiscard]] Mesh make_plane(float width = 1.0F, float depth = 1.0F);
[[nodiscard]] Mesh make_uv_sphere(float radius = 0.5F, int segments = 32, int rings = 16);
[[nodiscard]] Mesh make_cylinder(float radius = 0.5F, float height = 1.0F, int segments = 32);
[[nodiscard]] Mesh make_cone(float radius = 0.5F, float height = 1.0F, int segments = 32);

}  // namespace pacd::render
