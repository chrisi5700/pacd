//
// Analytic signed-distance functions for the solver's convex primitives, each
// evaluated in its local frame via its quaternion orientation. All are exact
// SDFs (negative inside, positive outside, |grad| = 1). A polynomial smooth-min
// composes them into the union field used for the final collision proxy.
//

#pragma once

#include <algorithm>
#include <cmath>
#include <type_traits>
#include <variant>

#include "pacd/solver/math.hpp"
#include "pacd/solver/primitives.hpp"

namespace pacd::solver
{

[[nodiscard]] inline float sd_sphere(Vec3 point, const Sphere& sphere) noexcept
{
	return length(point - sphere.pos) - sphere.radius;
}

[[nodiscard]] inline float sd_box(Vec3 point, const Box& box) noexcept
{
	const Vec3	local	= rotate_inverse(box.rot, point - box.pos);
	const Vec3	dvec	= abs(local) - (box.size * 0.5F);
	const Vec3	outside = max(dvec, Vec3{});
	const float inside	= std::min(std::max({dvec.x, dvec.y, dvec.z}), 0.0F);
	return length(outside) + inside;
}

[[nodiscard]] inline float sd_cylinder(Vec3 point, const Cylinder& cylinder) noexcept
{
	// Canonical cylinder: axis along local +Y, centred at the local origin.
	const Vec3	local  = rotate_inverse(cylinder.rot, point - cylinder.pos);
	const float radial = std::sqrt((local.x * local.x) + (local.z * local.z)) - cylinder.radius;
	const float axial  = std::abs(local.y) - (cylinder.height * 0.5F);
	const float out_r  = std::max(radial, 0.0F);
	const float out_a  = std::max(axial, 0.0F);
	const float inside = std::min(std::max(radial, axial), 0.0F);
	return std::sqrt((out_r * out_r) + (out_a * out_a)) + inside;
}

[[nodiscard]] inline float sd_primitive(Vec3 point, const Primitive& primitive) noexcept
{
	return std::visit(
		[point](const auto& shape) -> float
		{
			using Shape = std::decay_t<decltype(shape)>;
			if constexpr (std::is_same_v<Shape, Sphere>)
			{
				return sd_sphere(point, shape);
			}
			else if constexpr (std::is_same_v<Shape, Box>)
			{
				return sd_box(point, shape);
			}
			else
			{
				return sd_cylinder(point, shape);
			}
		},
		primitive);
}

// Polynomial smooth minimum: blends two SDFs so the union has continuous
// gradients. `blend` sets the blend radius; blend -> 0 recovers std::min.
[[nodiscard]] inline float smooth_min(float lhs, float rhs, float blend) noexcept
{
	if (blend <= 0.0F)
	{
		return std::min(lhs, rhs);
	}
	const float mix = std::clamp(0.5F + (0.5F * (rhs - lhs) / blend), 0.0F, 1.0F);
	return ((rhs * (1.0F - mix)) + (lhs * mix)) - (blend * mix * (1.0F - mix));
}

} // namespace pacd::solver
