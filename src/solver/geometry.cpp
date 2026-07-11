#include "pacd/solver/geometry.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>

namespace pacd::solver
{

// NOLINTBEGIN(readability-identifier-length) -- standard closest-point and
// solid-angle kernels; the short names mirror their textbook formulations.
namespace
{

// Closest point on triangle `tri` to `point` (Ericson, Real-Time Collision
// Detection, section 5.1.5).
[[nodiscard]] Vec3 closest_point_on_triangle(Vec3 point, const Tri& tri) noexcept
{
	const Vec3 ab = tri.b - tri.a;
	const Vec3 ac = tri.c - tri.a;
	const Vec3 ap = point - tri.a;

	const float d1 = dot(ab, ap);
	const float d2 = dot(ac, ap);
	if (d1 <= 0.0F && d2 <= 0.0F)
	{
		return tri.a;
	}

	const Vec3	bp = point - tri.b;
	const float d3 = dot(ab, bp);
	const float d4 = dot(ac, bp);
	if (d3 >= 0.0F && d4 <= d3)
	{
		return tri.b;
	}

	const Vec3	cp = point - tri.c;
	const float d5 = dot(ab, cp);
	const float d6 = dot(ac, cp);
	if (d6 >= 0.0F && d5 <= d6)
	{
		return tri.c;
	}

	const float vc = (d1 * d4) - (d3 * d2);
	if (vc <= 0.0F && d1 >= 0.0F && d3 <= 0.0F)
	{
		return tri.a + (ab * (d1 / (d1 - d3)));
	}

	const float vb = (d5 * d2) - (d1 * d6);
	if (vb <= 0.0F && d2 >= 0.0F && d6 <= 0.0F)
	{
		return tri.a + (ac * (d2 / (d2 - d6)));
	}

	const float va = (d3 * d6) - (d5 * d4);
	if (va <= 0.0F && (d4 - d3) >= 0.0F && (d5 - d6) >= 0.0F)
	{
		const float w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
		return tri.b + ((tri.c - tri.b) * w);
	}

	const float denom = 1.0F / (va + vb + vc);
	return tri.a + (ab * (vb * denom)) + (ac * (vc * denom));
}

// Signed solid angle subtended by `tri` at `point` (Van Oosterom & Strackee).
[[nodiscard]] float solid_angle(Vec3 point, const Tri& tri) noexcept
{
	const Vec3	pa = tri.a - point;
	const Vec3	pb = tri.b - point;
	const Vec3	pc = tri.c - point;
	const float la = length(pa);
	const float lb = length(pb);
	const float lc = length(pc);

	const float numerator = dot(pa, cross(pb, pc));
	const float denom	  = (la * lb * lc) + (dot(pa, pb) * lc) + (dot(pb, pc) * la) + (dot(pc, pa) * lb);
	return 2.0F * std::atan2(numerator, denom);
}

} // namespace
// NOLINTEND(readability-identifier-length)

float distance_point_triangle(Vec3 point, const Tri& tri) noexcept
{
	return length(point - closest_point_on_triangle(point, tri));
}

float winding_number(const TriMesh& mesh, Vec3 point) noexcept
{
	float total = 0.0F;
	for (std::size_t i = 0; i < mesh.triangle_count(); ++i)
	{
		total += solid_angle(point, mesh.triangle(i));
	}
	return total / (4.0F * PI_F);
}

float unsigned_distance(const TriMesh& mesh, Vec3 point) noexcept
{
	float best = std::numeric_limits<float>::max();
	for (std::size_t i = 0; i < mesh.triangle_count(); ++i)
	{
		best = std::min(best, distance_point_triangle(point, mesh.triangle(i)));
	}
	return best;
}

float signed_distance(const TriMesh& mesh, Vec3 point) noexcept
{
	const float dist = unsigned_distance(mesh, point);
	return (winding_number(mesh, point) > 0.5F) ? -dist : dist;
}

} // namespace pacd::solver
