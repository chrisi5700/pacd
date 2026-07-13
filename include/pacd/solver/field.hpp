//
// Signed distance field sampled on a regular grid over a mesh's padded bounding
// box. This is the oracle the fitter optimizes against: seeding reads its
// deepest interior node, and the inner loop integrates the "balloon" objective
// over points evaluated by trilinear interpolation.
//

#pragma once

#include <cstddef>
#include <vector>

#include "pacd/solver/math.hpp"
#include "pacd/solver/mesh.hpp"

namespace pacd::solver
{

// Grid of signed distances (negative inside). Node (cix, ciy, ciz) sits at
// origin + (cix, ciy, ciz) * cell; data is laid out with cix fastest, then ciy,
// then ciz.
struct DistanceField
{
	Vec3			   origin;
	Vec3			   cell{.x = 1.0F, .y = 1.0F, .z = 1.0F};
	int				   nx{0};
	int				   ny{0};
	int				   nz{0};
	std::vector<float> data;

	[[nodiscard]] std::size_t node_count() const noexcept
	{
		return static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny) * static_cast<std::size_t>(nz);
	}

	[[nodiscard]] std::size_t linear_index(int cix, int ciy, int ciz) const noexcept
	{
		return static_cast<std::size_t>(cix) +
			   (static_cast<std::size_t>(nx) *
				(static_cast<std::size_t>(ciy) + (static_cast<std::size_t>(ny) * static_cast<std::size_t>(ciz))));
	}

	[[nodiscard]] Vec3 node_position(int cix, int ciy, int ciz) const noexcept
	{
		return origin + vec3(static_cast<float>(cix) * cell.x, static_cast<float>(ciy) * cell.y,
							 static_cast<float>(ciz) * cell.z);
	}

	// World position of the node at a flat index (cix fastest, then ciy, then ciz):
	// the inverse of linear_index.
	[[nodiscard]] Vec3 node_position(std::size_t linear) const noexcept
	{
		const auto width = static_cast<std::size_t>(nx);
		const auto slice = width * static_cast<std::size_t>(ny);
		return node_position(static_cast<int>(linear % width), static_cast<int>((linear % slice) / width),
							 static_cast<int>(linear / slice));
	}

	[[nodiscard]] float at_index(int cix, int ciy, int ciz) const { return data.at(linear_index(cix, ciy, ciz)); }

	// Trilinearly interpolated signed distance at an arbitrary world point.
	// Points outside the grid clamp to the boundary node.
	[[nodiscard]] float sample(Vec3 point) const;
};

// Build a signed distance field for `mesh`. `resolution` = number of grid nodes
// along the longest padded-bbox axis; `padding` expands the bbox by this
// fraction of its diagonal so the field carries an exterior shell (needed by
// the optimizer's protrusion term).
[[nodiscard]] DistanceField build_distance_field(const TriMesh& mesh, int resolution, float padding);

} // namespace pacd::solver
