//
// Mesh geometry queries used to build the signed distance field: unsigned
// distance to the nearest triangle, and the generalized winding number that
// signs it. Brute-force (O(triangles) per query) -- correct and simple. The field
// build runs these through a BVH instead (see bvh.hpp) for ~O(log triangles); the
// brute versions here remain the exact reference the BVH is checked against, and
// still sign the thin band of nodes sitting right on the surface.
//

#pragma once

#include "pacd/solver/math.hpp"
#include "pacd/solver/mesh.hpp"

namespace pacd::solver
{

// Unsigned distance from `point` to a single triangle.
[[nodiscard]] float distance_point_triangle(Vec3 point, const Tri& tri) noexcept;

// Signed solid angle subtended by `tri` at `point` (Van Oosterom & Strackee). The
// per-triangle term of the generalized winding number, exposed so the BVH's fast
// winding number can sum it exactly in the near field.
[[nodiscard]] float solid_angle(Vec3 point, const Tri& tri) noexcept;

// Generalized winding number of `mesh` at `point`: the sum of signed solid
// angles / 4pi. ~1 deep inside a closed, outward-oriented mesh and ~0 outside;
// it degrades gracefully (fractional) on open / non-manifold input, which is
// why it -- not ray-parity -- signs the distance field.
[[nodiscard]] float winding_number(const TriMesh& mesh, Vec3 point) noexcept;

// Unsigned distance from `point` to the closest triangle of `mesh`.
[[nodiscard]] float unsigned_distance(const TriMesh& mesh, Vec3 point) noexcept;

// Signed distance to the mesh surface (negative inside); sign from the winding
// number thresholded at 0.5.
[[nodiscard]] float signed_distance(const TriMesh& mesh, Vec3 point) noexcept;

} // namespace pacd::solver
