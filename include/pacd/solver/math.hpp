//
// Minimal vector + quaternion math shared across the solver (primitives,
// meshes, SDFs). Kept separate from pacd::render::Vec3 so the solver has no
// render dependency.
//

#pragma once

#include <array>
#include <cmath>
#include <cstddef>
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

// Unit quaternion for the rotation whose columns -- the images of the local
// x/y/z axes -- are the given orthonormal, right-handed basis vectors. Shepperd's
// method, numerically stable across all orientations (picks the largest divisor).
[[nodiscard]] inline Quat quat_from_basis(Vec3 col_x, Vec3 col_y, Vec3 col_z) noexcept
{
	const float m00	  = col_x.x;
	const float m10	  = col_x.y;
	const float m20	  = col_x.z;
	const float m01	  = col_y.x;
	const float m11	  = col_y.y;
	const float m21	  = col_y.z;
	const float m02	  = col_z.x;
	const float m12	  = col_z.y;
	const float m22	  = col_z.z;
	const float trace = m00 + m11 + m22;

	Quat quat;
	if (trace > 0.0F)
	{
		const float scale = std::sqrt(trace + 1.0F) * 2.0F; // 4 * w
		quat = {.x = (m21 - m12) / scale, .y = (m02 - m20) / scale, .z = (m10 - m01) / scale, .w = 0.25F * scale};
	}
	else if (m00 > m11 && m00 > m22)
	{
		const float scale = std::sqrt(((1.0F + m00) - m11) - m22) * 2.0F; // 4 * x
		quat = {.x = 0.25F * scale, .y = (m01 + m10) / scale, .z = (m02 + m20) / scale, .w = (m21 - m12) / scale};
	}
	else if (m11 > m22)
	{
		const float scale = std::sqrt(((1.0F + m11) - m00) - m22) * 2.0F; // 4 * y
		quat = {.x = (m01 + m10) / scale, .y = 0.25F * scale, .z = (m12 + m21) / scale, .w = (m02 - m20) / scale};
	}
	else
	{
		const float scale = std::sqrt(((1.0F + m22) - m00) - m11) * 2.0F; // 4 * z
		quat = {.x = (m02 + m20) / scale, .y = (m12 + m21) / scale, .z = 0.25F * scale, .w = (m10 - m01) / scale};
	}
	return normalize(quat);
}

// Symmetric 3x3 matrix (row/column order x, y, z), stored by its upper triangle.
struct SymMat3
{
	float xx{0.0F};
	float yy{0.0F};
	float zz{0.0F};
	float xy{0.0F};
	float xz{0.0F};
	float yz{0.0F};
};

// Eigen-decomposition of a symmetric 3x3 matrix. `values` are in descending
// order; `vectors[i]` is the unit eigenvector for `values[i]`.
struct SymEigen
{
	std::array<float, 3> values{};
	std::array<Vec3, 3>	 vectors{};
};

namespace detail
{
using Mat3Rows = std::array<std::array<float, 3>, 3>;

// One symmetric Jacobi rotation that zeroes entry (row, col): updates the working
// matrix `aij` in place and accumulates the rotation into the eigenvector `basis`.
inline void jacobi_rotate(Mat3Rows& aij, Mat3Rows& basis, std::size_t row, std::size_t col)
{
	const float apq = aij.at(row).at(col);
	if (std::abs(apq) <= 1.0e-20F)
	{
		return;
	}
	const float tau	  = (aij.at(col).at(col) - aij.at(row).at(row)) / (2.0F * apq);
	const float sgn	  = (tau >= 0.0F) ? 1.0F : -1.0F;
	const float tan_t = sgn / (std::abs(tau) + std::sqrt((tau * tau) + 1.0F));
	const float cos_t = 1.0F / std::sqrt((tan_t * tan_t) + 1.0F);
	const float sin_t = tan_t * cos_t;
	for (std::size_t k = 0; k < 3; ++k) // rotate rows: a <- J^T a
	{
		const float a_row = aij.at(row).at(k);
		const float a_col = aij.at(col).at(k);
		aij.at(row).at(k) = (cos_t * a_row) - (sin_t * a_col);
		aij.at(col).at(k) = (sin_t * a_row) + (cos_t * a_col);
	}
	for (std::size_t k = 0; k < 3; ++k) // rotate columns of a and accumulate eigenvectors
	{
		const float a_row	= aij.at(k).at(row);
		const float a_col	= aij.at(k).at(col);
		aij.at(k).at(row)	= (cos_t * a_row) - (sin_t * a_col);
		aij.at(k).at(col)	= (sin_t * a_row) + (cos_t * a_col);
		const float v_row	= basis.at(k).at(row);
		const float v_col	= basis.at(k).at(col);
		basis.at(k).at(row) = (cos_t * v_row) - (sin_t * v_col);
		basis.at(k).at(col) = (sin_t * v_row) + (cos_t * v_col);
	}
}

// Assemble the eigenpairs from a (near-)diagonalised matrix, sorted descending.
[[nodiscard]] inline SymEigen order_eigen(const Mat3Rows& aij, const Mat3Rows& basis)
{
	std::array<std::size_t, 3> order = {0, 1, 2};
	const std::array<float, 3> diag	 = {aij.at(0).at(0), aij.at(1).at(1), aij.at(2).at(2)};
	for (std::size_t i = 0; i < 2; ++i)
	{
		for (std::size_t j = i + 1; j < 3; ++j)
		{
			if (diag.at(order.at(j)) > diag.at(order.at(i)))
			{
				const std::size_t tmp = order.at(i);
				order.at(i)			  = order.at(j);
				order.at(j)			  = tmp;
			}
		}
	}
	SymEigen out;
	for (std::size_t i = 0; i < 3; ++i)
	{
		const std::size_t src = order.at(i);
		out.values.at(i)	  = diag.at(src);
		out.vectors.at(i)	  = normalize(vec3(basis.at(0).at(src), basis.at(1).at(src), basis.at(2).at(src)));
	}
	return out;
}
} // namespace detail

// Diagonalise a symmetric 3x3 matrix with cyclic Jacobi rotations (a handful of
// sweeps converge to float precision for 3x3) -- dependency-free and enough for
// the solver's local shape analysis.
[[nodiscard]] inline SymEigen symmetric_eigen(const SymMat3& mat)
{
	using detail::Mat3Rows;
	Mat3Rows aij   = {std::array<float, 3>{mat.xx, mat.xy, mat.xz}, std::array<float, 3>{mat.xy, mat.yy, mat.yz},
					  std::array<float, 3>{mat.xz, mat.yz, mat.zz}};
	Mat3Rows basis = {std::array<float, 3>{1.0F, 0.0F, 0.0F}, std::array<float, 3>{0.0F, 1.0F, 0.0F},
					  std::array<float, 3>{0.0F, 0.0F, 1.0F}};

	constexpr int MAX_SWEEPS = 16;
	for (int sweep = 0; sweep < MAX_SWEEPS; ++sweep)
	{
		const float off = std::abs(aij.at(0).at(1)) + std::abs(aij.at(0).at(2)) + std::abs(aij.at(1).at(2));
		if (off <= 1.0e-12F)
		{
			break;
		}
		detail::jacobi_rotate(aij, basis, 0, 1);
		detail::jacobi_rotate(aij, basis, 0, 2);
		detail::jacobi_rotate(aij, basis, 1, 2);
	}
	return detail::order_eigen(aij, basis);
}

} // namespace pacd::solver
