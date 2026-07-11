//
// Lean triangle-mesh representation used as solver input: just positions and
// index triangles -- everything the mesh SDF (point-triangle distance +
// winding-number sign) and interior coverage sampling need, and nothing else.
// Deliberately decoupled from pacd::render::Mesh (no normals/materials/GL);
// convert loaded meshes with to_tri_mesh() from render_bridge.hpp.
//

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "pacd/solver/math.hpp"

namespace pacd::solver
{

// A triangle resolved to its three corner positions.
struct Tri
{
	Vec3 a;
	Vec3 b;
	Vec3 c;
};

using IndexTriangle = std::array<std::uint32_t, 3>;

struct TriMesh
{
	std::vector<Vec3>		   vertices;
	std::vector<IndexTriangle> triangles;

	[[nodiscard]] std::size_t vertex_count() const noexcept { return vertices.size(); }
	[[nodiscard]] std::size_t triangle_count() const noexcept { return triangles.size(); }

	// Resolve triangle `index` to its three corner positions.
	[[nodiscard]] Tri triangle(std::size_t index) const
	{
		const IndexTriangle& tri = triangles.at(index);
		return {.a = vertices.at(tri.at(0)), .b = vertices.at(tri.at(1)), .c = vertices.at(tri.at(2))};
	}
};

} // namespace pacd::solver
