#include "pacd/render/math.hpp"

#include <array>
#include <cmath>

namespace pacd::render
{

Vec3 transform_normal(const Mat4& model, Vec3 normal) noexcept
{
	// Upper-left 3x3 of the (column-major) model matrix.
	const std::array<float, 9> mat{model.at(0, 0), model.at(1, 0), model.at(2, 0), model.at(0, 1), model.at(1, 1),
								   model.at(2, 1), model.at(0, 2), model.at(1, 2), model.at(2, 2)};

	const float det = (mat.at(0) * ((mat.at(4) * mat.at(8)) - (mat.at(5) * mat.at(7)))) -
					  (mat.at(3) * ((mat.at(1) * mat.at(8)) - (mat.at(2) * mat.at(7)))) +
					  (mat.at(6) * ((mat.at(1) * mat.at(5)) - (mat.at(2) * mat.at(4))));

	if (std::abs(det) < 1e-12F)
	{
		return normalize(normal);
	}
	const float inv_det = 1.0F / det;

	// Cofactor (adjugate) entries of the 3x3, already giving inverse-transpose
	// when applied as normal * cofactor.
	const float c00 = ((mat.at(4) * mat.at(8)) - (mat.at(5) * mat.at(7))) * inv_det;
	const float c01 = ((mat.at(2) * mat.at(7)) - (mat.at(1) * mat.at(8))) * inv_det;
	const float c02 = ((mat.at(1) * mat.at(5)) - (mat.at(2) * mat.at(4))) * inv_det;
	const float c10 = ((mat.at(5) * mat.at(6)) - (mat.at(3) * mat.at(8))) * inv_det;
	const float c11 = ((mat.at(0) * mat.at(8)) - (mat.at(2) * mat.at(6))) * inv_det;
	const float c12 = ((mat.at(2) * mat.at(3)) - (mat.at(0) * mat.at(5))) * inv_det;
	const float c20 = ((mat.at(3) * mat.at(7)) - (mat.at(4) * mat.at(6))) * inv_det;
	const float c21 = ((mat.at(1) * mat.at(6)) - (mat.at(0) * mat.at(7))) * inv_det;
	const float c22 = ((mat.at(0) * mat.at(4)) - (mat.at(1) * mat.at(3))) * inv_det;

	const Vec3 out{(normal.x * c00) + (normal.y * c01) + (normal.z * c02),
				   (normal.x * c10) + (normal.y * c11) + (normal.z * c12),
				   (normal.x * c20) + (normal.y * c21) + (normal.z * c22)};
	return normalize(out);
}

} // namespace pacd::render
