#pragma once

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

} // namespace pacd::solver::test
