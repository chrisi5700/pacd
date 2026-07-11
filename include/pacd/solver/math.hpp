//
// Minimal vector + quaternion math shared across the solver (primitives,
// meshes, SDFs). Kept separate from pacd::render::Vec3 so the solver has no
// render dependency.
//

#pragma once

#include <cmath>
#include <numbers>

namespace pacd::solver
{

inline constexpr float PI_F = std::numbers::pi_v<float>;

struct Vec3
{
	float x{0.0F};
	float y{0.0F};
	float z{0.0F};
};

// Concise factory (the codebase uses designated-init aggregates, so this keeps
// call sites terse without a constructor).
[[nodiscard]] constexpr Vec3 vec3(float x, float y, float z) noexcept
{
	return {.x = x, .y = y, .z = z};
}

[[nodiscard]] constexpr Vec3 operator+(Vec3 lhs, Vec3 rhs) noexcept
{
	return {.x = lhs.x + rhs.x, .y = lhs.y + rhs.y, .z = lhs.z + rhs.z};
}

[[nodiscard]] constexpr Vec3 operator-(Vec3 lhs, Vec3 rhs) noexcept
{
	return {.x = lhs.x - rhs.x, .y = lhs.y - rhs.y, .z = lhs.z - rhs.z};
}

[[nodiscard]] constexpr Vec3 operator-(Vec3 vec) noexcept
{
	return {.x = -vec.x, .y = -vec.y, .z = -vec.z};
}

[[nodiscard]] constexpr Vec3 operator*(Vec3 vec, float scalar) noexcept
{
	return {.x = vec.x * scalar, .y = vec.y * scalar, .z = vec.z * scalar};
}

[[nodiscard]] constexpr Vec3 operator*(float scalar, Vec3 vec) noexcept
{
	return vec * scalar;
}

[[nodiscard]] constexpr float dot(Vec3 lhs, Vec3 rhs) noexcept
{
	return (lhs.x * rhs.x) + (lhs.y * rhs.y) + (lhs.z * rhs.z);
}

[[nodiscard]] constexpr Vec3 cross(Vec3 lhs, Vec3 rhs) noexcept
{
	return {.x = (lhs.y * rhs.z) - (lhs.z * rhs.y),
			.y = (lhs.z * rhs.x) - (lhs.x * rhs.z),
			.z = (lhs.x * rhs.y) - (lhs.y * rhs.x)};
}

[[nodiscard]] inline float length(Vec3 vec) noexcept
{
	return std::sqrt(dot(vec, vec));
}

[[nodiscard]] inline Vec3 normalize(Vec3 vec) noexcept
{
	const float len = length(vec);
	if (len <= 0.0F)
	{
		return {};
	}
	return vec * (1.0F / len);
}

// Unit quaternion stored as (x, y, z, w); identity is (0, 0, 0, 1).
struct Quat
{
	float x{0.0F};
	float y{0.0F};
	float z{0.0F};
	float w{1.0F};
};

// Hamilton product (compose rotations: `lhs` applied after `rhs`).
[[nodiscard]] constexpr Quat operator*(Quat lhs, Quat rhs) noexcept
{
	return {.x = (lhs.w * rhs.x) + (lhs.x * rhs.w) + (lhs.y * rhs.z) - (lhs.z * rhs.y),
			.y = (lhs.w * rhs.y) - (lhs.x * rhs.z) + (lhs.y * rhs.w) + (lhs.z * rhs.x),
			.z = (lhs.w * rhs.z) + (lhs.x * rhs.y) - (lhs.y * rhs.x) + (lhs.z * rhs.w),
			.w = (lhs.w * rhs.w) - (lhs.x * rhs.x) - (lhs.y * rhs.y) - (lhs.z * rhs.z)};
}

[[nodiscard]] constexpr Quat conjugate(Quat quat) noexcept
{
	return {.x = -quat.x, .y = -quat.y, .z = -quat.z, .w = quat.w};
}

[[nodiscard]] inline Quat normalize(Quat quat) noexcept
{
	const float len = std::sqrt((quat.x * quat.x) + (quat.y * quat.y) + (quat.z * quat.z) + (quat.w * quat.w));
	if (len <= 0.0F)
	{
		return {};
	}
	const float inv = 1.0F / len;
	return {.x = quat.x * inv, .y = quat.y * inv, .z = quat.z * inv, .w = quat.w * inv};
}

// Rotate `vec` by unit quaternion `quat` (v' = q v q*).
[[nodiscard]] inline Vec3 rotate(Quat quat, Vec3 vec) noexcept
{
	const Vec3 axis{.x = quat.x, .y = quat.y, .z = quat.z};
	const Vec3 tmp = cross(axis, vec) * 2.0F;
	return vec + (tmp * quat.w) + cross(axis, tmp);
}

// Rotate `vec` by the inverse of unit quaternion `quat` (world -> local).
[[nodiscard]] inline Vec3 rotate_inverse(Quat quat, Vec3 vec) noexcept
{
	return rotate(conjugate(quat), vec);
}

// Unit quaternion for a rotation of `angle` radians about `axis` (normalized).
[[nodiscard]] inline Quat quat_from_axis_angle(Vec3 axis, float angle) noexcept
{
	const Vec3	unit	 = normalize(axis);
	const float half	 = angle * 0.5F;
	const float sin_half = std::sin(half);
	return {.x = unit.x * sin_half, .y = unit.y * sin_half, .z = unit.z * sin_half, .w = std::cos(half)};
}

} // namespace pacd::solver
