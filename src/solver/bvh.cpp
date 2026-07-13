#include "pacd/solver/bvh.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <vector>

#include "pacd/solver/geometry.hpp"

namespace pacd::solver
{
namespace
{
constexpr std::size_t LEAF_SIZE	   = 4;			 // triangles per leaf
constexpr float	  WINDING_BETA = 2.0F;		 // far field: dipole when dist > BETA * subtree radius
constexpr float		  FOUR_PI	   = 4.0F * PI_F;

[[nodiscard]] float axis_component(Vec3 vec, int axis) noexcept
{
	if (axis == 0)
	{
		return vec.x;
	}
	if (axis == 1)
	{
		return vec.y;
	}
	return vec.z;
}

// Index (0/1/2) of the largest component, ties resolved to the lower axis.
[[nodiscard]] int widest_axis(Vec3 extent) noexcept
{
	int	  axis	 = 0;
	float widest = extent.x;
	if (extent.y > widest)
	{
		axis   = 1;
		widest = extent.y;
	}
	if (extent.z > widest)
	{
		axis = 2;
	}
	return axis;
}

// Squared distance from `point` to the axis-aligned box [low, high] (0 inside).
[[nodiscard]] float dist_sq_to_box(Vec3 point, Vec3 low, Vec3 high) noexcept
{
	const float gap_x = std::max({low.x - point.x, 0.0F, point.x - high.x});
	const float gap_y = std::max({low.y - point.y, 0.0F, point.y - high.y});
	const float gap_z = std::max({low.z - point.z, 0.0F, point.z - high.z});
	return (gap_x * gap_x) + (gap_y * gap_y) + (gap_z * gap_z);
}

} // namespace

TriBvh::TriBvh(const TriMesh& mesh)
{
	const std::size_t count = mesh.triangle_count();
	if (count == 0)
	{
		return;
	}

	// Per-triangle box and centroid, computed once; the tree is built over an index
	// permutation so the triangles themselves are copied only into the leaves.
	std::vector<Tri>  src(count);
	std::vector<Vec3> tri_low(count);
	std::vector<Vec3> tri_high(count);
	std::vector<Vec3> tri_center(count);
	for (std::size_t idx = 0; idx < count; ++idx)
	{
		const Tri tri	   = mesh.triangle(idx);
		src.at(idx)		   = tri;
		tri_low.at(idx)	   = min(min(tri.a, tri.b), tri.c);
		tri_high.at(idx)   = max(max(tri.a, tri.b), tri.c);
		tri_center.at(idx) = (tri.a + tri.b + tri.c) * (1.0F / 3.0F);
	}

	std::vector<std::size_t> order(count);
	std::ranges::iota(order, std::size_t{0});
	m_nodes.reserve(2 * count);
	m_tris.reserve(count);
	build(src, tri_low, tri_high, tri_center, order, 0, count);
}

TriBvh::Node TriBvh::make_leaf(const std::vector<Tri>& src, const std::vector<Vec3>& tri_center,
							   const std::vector<std::size_t>& order, std::size_t begin, std::size_t end, Vec3 low,
							   Vec3 high)
{
	Node leaf;
	leaf.low   = low;
	leaf.high  = high;
	leaf.first = m_tris.size();
	leaf.count = end - begin;

	Vec3  area_normal{};
	Vec3  weighted{};
	float area = 0.0F;
	for (std::size_t idx = begin; idx < end; ++idx)
	{
		const Tri&	tri		 = src.at(order.at(idx));
		const Vec3	vec_area = cross(tri.b - tri.a, tri.c - tri.a) * 0.5F; // oriented area, outward
		const float tri_area = length(vec_area);
		area_normal			 = area_normal + vec_area;
		weighted			 = weighted + (tri_center.at(order.at(idx)) * tri_area);
		area += tri_area;
		m_tris.push_back(tri);
	}
	leaf.area		 = area;
	leaf.area_normal = area_normal;
	leaf.center		 = (area > 0.0F) ? weighted * (1.0F / area) : tri_center.at(order.at(begin));

	float radius = 0.0F;
	for (std::size_t idx = begin; idx < end; ++idx)
	{
		const Tri& tri = src.at(order.at(idx));
		radius = std::max({radius, length(tri.a - leaf.center), length(tri.b - leaf.center), length(tri.c - leaf.center)});
	}
	leaf.radius = radius;
	return leaf;
}

TriBvh::Node TriBvh::combine(const Node& left, std::size_t left_idx, const Node& right, std::size_t right_idx, Vec3 low,
							 Vec3 high) noexcept
{
	Node node;
	node.low		 = low;
	node.high		 = high;
	node.left		 = left_idx;
	node.right		 = right_idx;
	node.count		 = 0;
	node.area		 = left.area + right.area;
	node.area_normal = left.area_normal + right.area_normal;
	node.center		 = (node.area > 0.0F)
						   ? ((left.center * left.area) + (right.center * right.area)) * (1.0F / node.area)
						   : left.center;
	node.radius = std::max(length(left.center - node.center) + left.radius, length(right.center - node.center) + right.radius);
	return node;
}

std::size_t TriBvh::build(const std::vector<Tri>& src, const std::vector<Vec3>& tri_low,
						  const std::vector<Vec3>& tri_high, const std::vector<Vec3>& tri_center,
						  std::vector<std::size_t>& order, std::size_t begin, std::size_t end)
{
	const std::size_t node_idx = m_nodes.size();
	m_nodes.push_back(Node{}); // reserve this node's slot; children take higher indices

	Vec3 low  = tri_low.at(order.at(begin));
	Vec3 high = tri_high.at(order.at(begin));
	for (std::size_t idx = begin + 1; idx < end; ++idx)
	{
		low	 = min(low, tri_low.at(order.at(idx)));
		high = max(high, tri_high.at(order.at(idx)));
	}

	if (end - begin <= LEAF_SIZE)
	{
		m_nodes.at(node_idx) = make_leaf(src, tri_center, order, begin, end, low, high);
		return node_idx;
	}

	// Split at the median centroid along the widest axis of the centroid bounds.
	Vec3 centroid_low  = tri_center.at(order.at(begin));
	Vec3 centroid_high = centroid_low;
	for (std::size_t idx = begin + 1; idx < end; ++idx)
	{
		centroid_low  = min(centroid_low, tri_center.at(order.at(idx)));
		centroid_high = max(centroid_high, tri_center.at(order.at(idx)));
	}
	const int		  axis = widest_axis(centroid_high - centroid_low);
	const std::size_t mid  = begin + ((end - begin) / 2);
	std::nth_element(order.begin() + static_cast<std::ptrdiff_t>(begin), order.begin() + static_cast<std::ptrdiff_t>(mid),
					 order.begin() + static_cast<std::ptrdiff_t>(end), [&](std::size_t lhs, std::size_t rhs)
					 { return axis_component(tri_center.at(lhs), axis) < axis_component(tri_center.at(rhs), axis); });

	const std::size_t left	= build(src, tri_low, tri_high, tri_center, order, begin, mid);
	const std::size_t right = build(src, tri_low, tri_high, tri_center, order, mid, end);
	m_nodes.at(node_idx)	= combine(m_nodes.at(left), left, m_nodes.at(right), right, low, high);
	return node_idx;
}

float TriBvh::unsigned_distance(Vec3 point) const noexcept
{
	if (m_nodes.empty())
	{
		return std::numeric_limits<float>::max();
	}
	float best_sq = std::numeric_limits<float>::max();
	nearest(0, point, best_sq);
	return std::sqrt(best_sq);
}

void TriBvh::nearest(std::size_t node_idx, Vec3 point, float& best_sq) const noexcept
{
	const Node& node = m_nodes.at(node_idx);
	if (dist_sq_to_box(point, node.low, node.high) >= best_sq)
	{
		return; // the whole box is farther than the best triangle found so far
	}
	if (node.count > 0)
	{
		for (std::size_t idx = 0; idx < node.count; ++idx)
		{
			const float dist = distance_point_triangle(point, m_tris.at(node.first + idx));
			best_sq			 = std::min(best_sq, dist * dist);
		}
		return;
	}
	const float left_sq	 = dist_sq_to_box(point, m_nodes.at(node.left).low, m_nodes.at(node.left).high);
	const float right_sq = dist_sq_to_box(point, m_nodes.at(node.right).low, m_nodes.at(node.right).high);
	if (left_sq <= right_sq) // nearer child first -> tighter best_sq sooner -> more pruning
	{
		nearest(node.left, point, best_sq);
		nearest(node.right, point, best_sq);
	}
	else
	{
		nearest(node.right, point, best_sq);
		nearest(node.left, point, best_sq);
	}
}

float TriBvh::winding_number(Vec3 point) const noexcept
{
	if (m_nodes.empty())
	{
		return 0.0F;
	}
	float omega = 0.0F;
	accumulate(0, point, omega);
	return omega / FOUR_PI;
}

void TriBvh::accumulate(std::size_t node_idx, Vec3 point, float& omega) const noexcept
{
	const Node& node   = m_nodes.at(node_idx);
	const Vec3	offset = node.center - point;
	const float dist   = length(offset);
	if (dist > WINDING_BETA * node.radius) // far field: one dipole term for the whole subtree
	{
		omega += dot(node.area_normal, offset) / (dist * dist * dist);
		return;
	}
	if (node.count > 0) // near a leaf: exact per-triangle solid angles
	{
		for (std::size_t idx = 0; idx < node.count; ++idx)
		{
			omega += solid_angle(point, m_tris.at(node.first + idx));
		}
		return;
	}
	accumulate(node.left, point, omega);
	accumulate(node.right, point, omega);
}

} // namespace pacd::solver
