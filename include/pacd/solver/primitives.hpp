//
// Created by chris on 7/11/26.
//

#ifndef PACD_PRIMITIVES_HPP
#define PACD_PRIMITIVES_HPP

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

struct Primitive : PrimitiveBase
{
	[[nodiscard]] Vec3 get_pos() const
	{
		return std::visit([](const auto& primitive) { return primitive.pos; }, *this);
	}
};

} // namespace pacd::solver

#endif // PACD_PRIMITIVES_HPP
