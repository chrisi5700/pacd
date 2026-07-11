#pragma once

#include <algorithm>
#include <cstdint>
#include <iterator>

#include "pacd/solver/math.hpp"
#include "pacd/solver/mesh.hpp"

namespace pacd::solver::test
{

// Axis-aligned cube centred at the origin, half-extent `half`, with 12
// consistently outward-oriented triangles (winding number ~= +1 inside).
[[nodiscard]] inline TriMesh make_box_mesh(float half)
{
	TriMesh mesh;
	mesh.vertices  = {vec3(-half, -half, -half), vec3(half, -half, -half), vec3(half, half, -half),
					  vec3(-half, half, -half),	 vec3(-half, -half, half), vec3(half, -half, half),
					  vec3(half, half, half),	 vec3(-half, half, half)};
	mesh.triangles = {
		{1, 2, 6}, {1, 6, 5}, // +X
		{0, 4, 7}, {0, 7, 3}, // -X
		{3, 7, 6}, {3, 6, 2}, // +Y
		{0, 1, 5}, {0, 5, 4}, // -Y
		{4, 5, 6}, {4, 6, 7}, // +Z
		{0, 3, 2}, {0, 2, 1}, // -Z
	};
	return mesh;
}

// Oriented rectangular beam centred at the origin: the unit cube scaled to the
// per-axis half-extents `half`, then rotated by `rot`. Scaling by positive
// extents and rotating both preserve the outward winding of make_box_mesh.
[[nodiscard]] inline TriMesh make_beam_mesh(Vec3 half, Quat rot)
{
	TriMesh mesh = make_box_mesh(1.0F);
	std::ranges::transform(mesh.vertices, mesh.vertices.begin(), [half, rot](Vec3 vert)
						   { return rotate(rot, vec3(vert.x * half.x, vert.y * half.y, vert.z * half.z)); });
	return mesh;
}

// Two axis-aligned cubes of half-extent `half`, centred at (+offset, 0, 0) and
// (-offset, 0, 0). With offset > half the cubes are disjoint and the shape has an
// exact mirror symmetry across x = 0 (plus the cube symmetries of each lobe).
[[nodiscard]] inline TriMesh make_two_box_mesh(float half, float offset)
{
	TriMesh		  mesh;
	const TriMesh unit = make_box_mesh(half);
	for (const float shift : {offset, -offset})
	{
		const auto base = static_cast<std::uint32_t>(mesh.vertices.size());
		std::ranges::transform(unit.vertices, std::back_inserter(mesh.vertices),
							   [shift](const Vec3& vert) { return vec3(vert.x + shift, vert.y, vert.z); });
		std::ranges::transform(unit.triangles, std::back_inserter(mesh.triangles), [base](const IndexTriangle& tri)
							   { return IndexTriangle{tri.at(0) + base, tri.at(1) + base, tri.at(2) + base}; });
	}
	return mesh;
}

} // namespace pacd::solver::test
