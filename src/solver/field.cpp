#include "pacd/solver/field.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <thread>
#include <vector>

#include "pacd/solver/bvh.hpp"
#include "pacd/solver/geometry.hpp"

namespace pacd::solver
{

// Half-width of the inside/outside boundary band around winding 0.5 within which the
// fast winding number is replaced by the exact one, so on-surface nodes are signed
// identically to the reference field. Comfortably wider than the fast winding's
// far-field error, yet far from the ~0 / ~1 winding of genuine exterior / interior.
constexpr float WINDING_BOUNDARY_BAND = 0.05F;

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
		min_pt = min(min_pt, vertex);
		max_pt = max(max_pt, vertex);
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

	// One BVH accelerates both halves of each node's signed distance -- the nearest
	// triangle and the winding-number sign -- from O(triangles) to ~O(log triangles).
	// It is built once and queried read-only, so all workers share it.
	const TriBvh bvh(mesh);

	// Each node's signed distance depends only on its own position -- no shared
	// state, no ordering -- so the grid fills in parallel with zero synchronisation:
	// contiguous flat-index ranges are handed to worker threads that write disjoint
	// slots. The result matches a serial fill; parallelism only changes wall-clock.
	const std::size_t total = field.node_count();
	const auto		  cores = static_cast<std::size_t>(std::max(1U, std::thread::hardware_concurrency()));
	const std::size_t workers = std::max<std::size_t>(1, std::min(cores, total));

	// The fast winding number is approximate, and its tree-ordered sum rounds a hair
	// differently from the brute sum right at the 0.5 inside/outside boundary -- which
	// only matters for the few nodes sitting almost exactly on the surface. For those
	// (a thin O(surface) band, not the O(volume) interior) recompute the exact winding
	// so the sign matches the reference field node-for-node; everywhere else the fast
	// winding is ~0 or ~1, nowhere near the boundary.
	const auto fill_range = [&bvh, &mesh, &field](std::size_t begin, std::size_t end)
	{
		for (std::size_t node = begin; node < end; ++node)
		{
			const Vec3	point	 = field.node_position(node);
			const float distance = bvh.unsigned_distance(point);
			float		winding	 = bvh.winding_number(point);
			if (std::abs(winding - 0.5F) < WINDING_BOUNDARY_BAND)
			{
				winding = winding_number(mesh, point);
			}
			field.data.at(node) = (winding > 0.5F) ? -distance : distance;
		}
	};

	const std::size_t		 chunk = (total + workers - 1) / workers;
	std::vector<std::thread> pool;
	pool.reserve(workers - 1);
	for (std::size_t slot = 1; slot < workers; ++slot)
	{
		const std::size_t begin = std::min(slot * chunk, total);
		pool.emplace_back(fill_range, begin, std::min(begin + chunk, total));
	}
	fill_range(0, std::min(chunk, total)); // this thread takes the first chunk
	for (std::thread& worker : pool)
	{
		worker.join();
	}
	return field;
}

} // namespace pacd::solver
