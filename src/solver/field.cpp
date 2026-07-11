#include "pacd/solver/field.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>

#include "pacd/solver/geometry.hpp"

namespace pacd::solver
{

float DistanceField::sample(Vec3 point) const
{
	// NOLINTBEGIN(readability-identifier-length) -- trilinear interpolation kernel.
	const float grid_x = std::clamp((point.x - origin.x) / cell.x, 0.0F, static_cast<float>(nx - 1));
	const float grid_y = std::clamp((point.y - origin.y) / cell.y, 0.0F, static_cast<float>(ny - 1));
	const float grid_z = std::clamp((point.z - origin.z) / cell.z, 0.0F, static_cast<float>(nz - 1));

	const int lo_i = static_cast<int>(std::floor(grid_x));
	const int lo_j = static_cast<int>(std::floor(grid_y));
	const int lo_k = static_cast<int>(std::floor(grid_z));
	const int hi_i = std::min(lo_i + 1, nx - 1);
	const int hi_j = std::min(lo_j + 1, ny - 1);
	const int hi_k = std::min(lo_k + 1, nz - 1);

	const float fx = grid_x - static_cast<float>(lo_i);
	const float fy = grid_y - static_cast<float>(lo_j);
	const float fz = grid_z - static_cast<float>(lo_k);

	const float c000 = at_index(lo_i, lo_j, lo_k);
	const float c100 = at_index(hi_i, lo_j, lo_k);
	const float c010 = at_index(lo_i, hi_j, lo_k);
	const float c110 = at_index(hi_i, hi_j, lo_k);
	const float c001 = at_index(lo_i, lo_j, hi_k);
	const float c101 = at_index(hi_i, lo_j, hi_k);
	const float c011 = at_index(lo_i, hi_j, hi_k);
	const float c111 = at_index(hi_i, hi_j, hi_k);

	const float c00 = c000 + ((c100 - c000) * fx);
	const float c01 = c001 + ((c101 - c001) * fx);
	const float c10 = c010 + ((c110 - c010) * fx);
	const float c11 = c011 + ((c111 - c011) * fx);
	const float c0	= c00 + ((c10 - c00) * fy);
	const float c1	= c01 + ((c11 - c01) * fy);
	return c0 + ((c1 - c0) * fz);
	// NOLINTEND(readability-identifier-length)
}

DistanceField build_distance_field(const TriMesh& mesh, int resolution, float padding)
{
	Vec3 min_pt = mesh.vertices.front();
	Vec3 max_pt = min_pt;
	for (const Vec3& vertex : mesh.vertices)
	{
		min_pt = vec3(std::min(min_pt.x, vertex.x), std::min(min_pt.y, vertex.y), std::min(min_pt.z, vertex.z));
		max_pt = vec3(std::max(max_pt.x, vertex.x), std::max(max_pt.y, vertex.y), std::max(max_pt.z, vertex.z));
	}

	const Vec3	span	= max_pt - min_pt;
	const float pad		= padding * length(span);
	const Vec3	padded	= vec3(span.x + (2.0F * pad), span.y + (2.0F * pad), span.z + (2.0F * pad));
	const float longest = std::max({padded.x, padded.y, padded.z});
	const float spacing = longest / static_cast<float>(std::max(resolution, 1));

	DistanceField field;
	field.origin = min_pt - vec3(pad, pad, pad);
	field.cell	 = vec3(spacing, spacing, spacing);
	field.nx	 = std::max(2, static_cast<int>(std::ceil(padded.x / spacing)) + 1);
	field.ny	 = std::max(2, static_cast<int>(std::ceil(padded.y / spacing)) + 1);
	field.nz	 = std::max(2, static_cast<int>(std::ceil(padded.z / spacing)) + 1);
	field.data.resize(field.node_count());

	for (int k = 0; k < field.nz; ++k)
	{
		for (int j = 0; j < field.ny; ++j)
		{
			for (int i = 0; i < field.nx; ++i)
			{
				field.data.at(field.linear_index(i, j, k)) = signed_distance(mesh, field.node_position(i, j, k));
			}
		}
	}
	return field;
}

} // namespace pacd::solver
