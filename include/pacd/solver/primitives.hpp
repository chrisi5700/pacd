//
// Created by chris on 7/11/26.
//

#ifndef PACD_PRIMITIVES_HPP
#define PACD_PRIMITIVES_HPP

#include <algorithm>
#include <cmath>
#include <variant>

#include "pacd/solver/math.hpp"

namespace pacd::solver
{

struct Box
{
	Vec3 pos;
	Vec3 size; // full extents along the local axes
	Quat rot;  // local -> world orientation
};

struct Sphere
{
	Vec3  pos;
	float radius{0.0F};
	// no orientation: a sphere is rotationally symmetric
};

struct Cylinder
{
	Vec3  pos;
	float radius{0.0F};
	float height{0.0F}; // full height along the local +Y axis
	Quat  rot;			// local -> world orientation
};

using PrimitiveBase = std::variant<Box, Sphere, Cylinder>;

// The variant is inherited (rather than held) so that operations common to all
// three shapes read as member functions on the primitive itself. Shape-specific
// analytic SDFs stay free functions in primitive_sdf.hpp -- keeping the field math
// out of the data type -- but pure geometric queries belong here.
struct Primitive : PrimitiveBase
{
	// World-space centre (every shape stores its centre in `pos`).
	[[nodiscard]] Vec3 center() const
	{
		return std::visit([](const auto& shape) { return shape.pos; }, *this);
	}

	// Local -> world orientation; identity for a sphere, which is rotationally
	// symmetric and carries no `rot`.
	[[nodiscard]] Quat rotation() const
	{
		if (const auto* box = std::get_if<Box>(this))
		{
			return box->rot;
		}
		if (const auto* cyl = std::get_if<Cylinder>(this))
		{
			return cyl->rot;
		}
		return {};
	}

	// Radius of the smallest sphere about `center()` that contains the primitive.
	[[nodiscard]] float bounding_radius() const
	{
		if (const auto* sphere = std::get_if<Sphere>(this))
		{
			return sphere->radius;
		}
		if (const auto* box = std::get_if<Box>(this))
		{
			return 0.5F * length(box->size);
		}
		const auto& cyl = std::get<Cylinder>(*this);
		return std::sqrt((cyl.radius * cyl.radius) + (0.25F * cyl.height * cyl.height));
	}

	// World-space direction of the primitive's longest axis (a cylinder's axis, a
	// box's widest side); a zero vector for a sphere, which has no long axis.
	[[nodiscard]] Vec3 long_axis() const
	{
		if (const auto* cyl = std::get_if<Cylinder>(this))
		{
			return rotate(cyl->rot, vec3(0.0F, 1.0F, 0.0F));
		}
		if (const auto* box = std::get_if<Box>(this))
		{
			Vec3 local = vec3(1.0F, 0.0F, 0.0F);
			if (box->size.y >= box->size.x && box->size.y >= box->size.z)
			{
				local = vec3(0.0F, 1.0F, 0.0F);
			}
			else if (box->size.z >= box->size.x && box->size.z >= box->size.y)
			{
				local = vec3(0.0F, 0.0F, 1.0F);
			}
			return rotate(box->rot, local);
		}
		return vec3(0.0F, 0.0F, 0.0F);
	}

	// World-space extent along `long_axis()`: a cylinder's height, a box's widest
	// side, a sphere's diameter.
	[[nodiscard]] float long_extent() const
	{
		if (const auto* cyl = std::get_if<Cylinder>(this))
		{
			return cyl->height;
		}
		if (const auto* box = std::get_if<Box>(this))
		{
			return std::max({box->size.x, box->size.y, box->size.z});
		}
		return 2.0F * std::get<Sphere>(*this).radius;
	}

	// Lower-case shape name for logging and diagnostics.
	[[nodiscard]] const char* kind_name() const
	{
		if (std::holds_alternative<Sphere>(*this))
		{
			return "sphere";
		}
		if (std::holds_alternative<Box>(*this))
		{
			return "box";
		}
		return "cylinder";
	}
};

} // namespace pacd::solver

#endif // PACD_PRIMITIVES_HPP
