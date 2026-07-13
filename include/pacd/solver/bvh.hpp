//
// One bounding-volume hierarchy over a mesh's triangles, serving both halves of
// the signed-distance field build in ~O(log triangles) instead of O(triangles):
//
//   * unsigned_distance -- exact nearest-triangle distance by branch-and-bound:
//     descend the nearer child first and prune any box farther than the best hit.
//
//   * winding_number -- the generalized winding number by a Barnes-Hut sum: a
//     subtree far from the query contributes a single dipole term from its
//     aggregated oriented area; close to the surface it descends to exact
//     per-triangle solid angles. It stays robust on arbitrary (open, non-manifold)
//     input -- the same generalized winding number, only approximated where the
//     query is far enough that the approximation cannot change the sign.
//
// Built once per mesh and queried read-only, so build_distance_field shares a
// single tree across every grid node and worker thread.
//

#pragma once

#include <cstddef>
#include <vector>

#include "pacd/solver/math.hpp"
#include "pacd/solver/mesh.hpp"

namespace pacd::solver
{

class TriBvh
{
public:
	explicit TriBvh(const TriMesh& mesh);

	// Unsigned distance to the nearest triangle -- the same value the brute-force
	// scan returns. A large sentinel for an empty tree.
	[[nodiscard]] float unsigned_distance(Vec3 point) const noexcept;

	// Generalized winding number (~1 inside a closed outward mesh, ~0 outside).
	[[nodiscard]] float winding_number(Vec3 point) const noexcept;

	[[nodiscard]] bool empty() const noexcept { return m_nodes.empty(); }

private:
	struct Node
	{
		Vec3		low;			 // AABB minimum corner (distance pruning)
		Vec3		high;			 // AABB maximum corner
		Vec3		area_normal;	 // sum of triangle oriented areas (winding dipole)
		Vec3		center;			 // area-weighted centroid of the subtree
		float		area{0.0F};		 // total unsigned triangle area of the subtree
		float		radius{0.0F};	 // max distance from `center` to any triangle vertex below
		std::size_t left{0};		 // child node index (internal only)
		std::size_t right{0};		 // child node index (internal only)
		std::size_t first{0};		 // leaf: index of its first triangle in m_tris
		std::size_t count{0};		 // leaf: triangle count; 0 marks an internal node
	};

	std::size_t build(const std::vector<Tri>& src, const std::vector<Vec3>& tri_low,
					  const std::vector<Vec3>& tri_high, const std::vector<Vec3>& tri_center,
					  std::vector<std::size_t>& order, std::size_t begin, std::size_t end);
	// Leaf node over triangles order[begin, end): copies them into m_tris and
	// aggregates the winding-number data (oriented area, area-weighted centre, radius).
	Node make_leaf(const std::vector<Tri>& src, const std::vector<Vec3>& tri_center,
				   const std::vector<std::size_t>& order, std::size_t begin, std::size_t end, Vec3 low, Vec3 high);
	// Internal node combining two already-built children (union AABB, summed area
	// and oriented area, area-weighted centre, conservative radius).
	[[nodiscard]] static Node combine(const Node& left, std::size_t left_idx, const Node& right, std::size_t right_idx,
									  Vec3 low, Vec3 high) noexcept;
	void nearest(std::size_t node, Vec3 point, float& best_sq) const noexcept;
	void accumulate(std::size_t node, Vec3 point, float& omega) const noexcept;

	std::vector<Node> m_nodes;
	std::vector<Tri>  m_tris; // triangles reordered so each leaf owns a contiguous run
};

} // namespace pacd::solver
