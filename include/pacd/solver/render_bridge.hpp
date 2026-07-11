//
// Optional glue: convert a loaded pacd::render::Mesh into the solver's TriMesh.
// Kept in its own header so the core solver stays free of any render/GL
// dependency -- only translation units that actually load meshes include this.
//

#pragma once

#include "pacd/render/mesh.hpp"
#include "pacd/solver/mesh.hpp"

namespace pacd::solver
{

[[nodiscard]] inline TriMesh to_tri_mesh(const render::Mesh& mesh)
{
	TriMesh out;
	out.vertices.reserve(mesh.positions.size());
	for (const render::Vec3& position : mesh.positions)
	{
		out.vertices.push_back(vec3(position.x, position.y, position.z));
	}
	out.triangles = mesh.triangles; // both are std::vector<std::array<uint32_t, 3>>
	return out;
}

} // namespace pacd::solver
