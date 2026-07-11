#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <variant>

#include "pacd/solver/field.hpp"
#include "pacd/solver/geometry.hpp"
#include "pacd/solver/math.hpp"
#include "pacd/solver/primitive_sdf.hpp"
#include "pacd/solver/primitives.hpp"
#include "pacd/solver/solver.hpp"
#include "solver_fixtures.hpp"

using namespace pacd::solver;
using Catch::Approx;

namespace
{
constexpr float EPS = 1.0e-4F;
}

TEST_CASE("quaternion rotates a vector about an axis", "[math]")
{
	const Quat rot	   = quat_from_axis_angle(vec3(0.0F, 0.0F, 1.0F), PI_F * 0.5F);
	const Vec3 rotated = rotate(rot, vec3(1.0F, 0.0F, 0.0F));
	REQUIRE(rotated.x == Approx(0.0F).margin(EPS));
	REQUIRE(rotated.y == Approx(1.0F).margin(EPS));
	REQUIRE(rotated.z == Approx(0.0F).margin(EPS));
}

TEST_CASE("quaternion inverse rotation round-trips", "[math]")
{
	const Quat rot	 = quat_from_axis_angle(vec3(0.3F, 1.0F, -0.2F), 0.9F);
	const Vec3 point = vec3(0.4F, -1.1F, 2.0F);
	const Vec3 back	 = rotate_inverse(rot, rotate(rot, point));
	REQUIRE(back.x == Approx(point.x).margin(EPS));
	REQUIRE(back.y == Approx(point.y).margin(EPS));
	REQUIRE(back.z == Approx(point.z).margin(EPS));
}

TEST_CASE("sphere SDF matches distance to centre", "[sdf]")
{
	const Sphere sphere{.pos = vec3(0.0F, 0.0F, 0.0F), .radius = 1.0F};
	REQUIRE(sd_sphere(vec3(0.0F, 0.0F, 0.0F), sphere) == Approx(-1.0F));
	REQUIRE(sd_sphere(vec3(1.0F, 0.0F, 0.0F), sphere) == Approx(0.0F).margin(EPS));
	REQUIRE(sd_sphere(vec3(3.0F, 0.0F, 0.0F), sphere) == Approx(2.0F));
}

TEST_CASE("box SDF respects orientation", "[sdf]")
{
	const Vec3 probe = vec3(0.0F, 0.5F, 0.0F);

	// Thin slab: 0.2 thick along local Y. The probe sits 0.4 outside the top face.
	const Box flat{.pos = vec3(0.0F, 0.0F, 0.0F), .size = vec3(4.0F, 0.2F, 4.0F), .rot = {}};
	REQUIRE(sd_box(probe, flat) == Approx(0.4F).margin(EPS));

	// Rotate 90 deg about Z: the thin axis moves to world X, so the same probe is
	// now inside the slab.
	const Box tilted{.pos  = vec3(0.0F, 0.0F, 0.0F),
					 .size = vec3(4.0F, 0.2F, 4.0F),
					 .rot  = quat_from_axis_angle(vec3(0.0F, 0.0F, 1.0F), PI_F * 0.5F)};
	REQUIRE(sd_box(probe, tilted) == Approx(-0.1F).margin(EPS));
}

TEST_CASE("cylinder SDF respects orientation", "[sdf]")
{
	// Long thin cylinder: radius 0.5, height 4 along local +Y.
	const Cylinder upright{.pos = vec3(0.0F, 0.0F, 0.0F), .radius = 0.5F, .height = 4.0F, .rot = {}};
	REQUIRE(sd_cylinder(vec3(1.0F, 0.0F, 0.0F), upright) == Approx(0.5F).margin(EPS));
	REQUIRE(sd_cylinder(vec3(0.0F, 1.0F, 0.0F), upright) == Approx(-0.5F).margin(EPS));

	// Rotate 90 deg about Z: axis now runs along world X, so the two probes swap
	// their inside/outside roles.
	const Cylinder lying{.pos	 = vec3(0.0F, 0.0F, 0.0F),
						 .radius = 0.5F,
						 .height = 4.0F,
						 .rot	 = quat_from_axis_angle(vec3(0.0F, 0.0F, 1.0F), PI_F * 0.5F)};
	REQUIRE(sd_cylinder(vec3(1.0F, 0.0F, 0.0F), lying) == Approx(-0.5F).margin(EPS));
	REQUIRE(sd_cylinder(vec3(0.0F, 1.0F, 0.0F), lying) == Approx(0.5F).margin(EPS));
}

TEST_CASE("smooth_min degrades to min and never exceeds its inputs", "[sdf]")
{
	REQUIRE(smooth_min(0.3F, 0.7F, 0.0F) == Approx(0.3F));
	REQUIRE(smooth_min(1.0F, 1.0F, 0.2F) <= 1.0F);
	REQUIRE(smooth_min(-2.0F, 5.0F, 0.1F) == Approx(-2.0F).margin(0.05F));
}

TEST_CASE("point-triangle distance projects onto the face interior", "[geometry]")
{
	const Tri tri{.a = vec3(0.0F, 0.0F, 0.0F), .b = vec3(1.0F, 0.0F, 0.0F), .c = vec3(0.0F, 1.0F, 0.0F)};
	REQUIRE(distance_point_triangle(vec3(0.25F, 0.25F, 2.0F), tri) == Approx(2.0F).margin(EPS));
	REQUIRE(distance_point_triangle(vec3(0.25F, 0.25F, 0.0F), tri) == Approx(0.0F).margin(EPS));
}

TEST_CASE("winding number is ~1 inside a closed mesh and ~0 outside", "[geometry]")
{
	const TriMesh cube = test::make_box_mesh(1.0F);
	REQUIRE(winding_number(cube, vec3(0.0F, 0.0F, 0.0F)) == Approx(1.0F).margin(1.0e-3F));
	REQUIRE(winding_number(cube, vec3(5.0F, 0.0F, 0.0F)) == Approx(0.0F).margin(1.0e-3F));
	REQUIRE(winding_number(cube, vec3(0.7F, -0.3F, 0.2F)) == Approx(1.0F).margin(1.0e-3F));
}

TEST_CASE("signed distance is negative inside, positive outside", "[geometry]")
{
	const TriMesh cube = test::make_box_mesh(1.0F);
	REQUIRE(signed_distance(cube, vec3(0.0F, 0.0F, 0.0F)) == Approx(-1.0F).margin(EPS));
	REQUIRE(signed_distance(cube, vec3(1.5F, 0.0F, 0.0F)) == Approx(0.5F).margin(EPS));
	REQUIRE(signed_distance(cube, vec3(0.0F, 0.0F, -1.25F)) == Approx(0.25F).margin(EPS));
}

TEST_CASE("distance field trilinearly approximates the mesh SDF", "[field]")
{
	const TriMesh		cube  = test::make_box_mesh(1.0F);
	const DistanceField field = build_distance_field(cube, 32, 0.1F);
	REQUIRE(field.sample(vec3(0.0F, 0.0F, 0.0F)) == Approx(-1.0F).margin(0.12F));
	REQUIRE(field.sample(vec3(1.2F, 0.0F, 0.0F)) == Approx(0.2F).margin(0.12F)); // within the padded shell
	REQUIRE(field.sample(vec3(0.0F, 0.0F, 0.0F)) < 0.0F);
	REQUIRE(field.sample(vec3(2.0F, 2.0F, 2.0F)) > 0.0F);
}

namespace
{
[[nodiscard]] SolverConfig fast_config()
{
	SolverConfig config;
	config.sample_count	  = 800;
	config.gd_iterations  = 50;
	config.sdf_resolution = 18;
	return config;
}
} // namespace

TEST_CASE("decompose fills a cube with a single oriented box", "[decompose]")
{
	const TriMesh				 cube  = test::make_box_mesh(1.0F);
	const std::vector<Primitive> parts = decompose(cube, fast_config());

	REQUIRE(!parts.empty());
	REQUIRE(parts.size() <= 6);

	// A box is the best convex fit for a cube, so it should win the first slot and
	// roughly reproduce the cube (size 2 per axis, covering the centre).
	REQUIRE(std::holds_alternative<Box>(parts.front()));
	const Box& box = std::get<Box>(parts.front());
	REQUIRE(sd_box(vec3(0.0F, 0.0F, 0.0F), box) < 0.0F);
	REQUIRE(box.size.x == Approx(2.0F).margin(0.6F));
	REQUIRE(box.size.y == Approx(2.0F).margin(0.6F));
	REQUIRE(box.size.z == Approx(2.0F).margin(0.6F));
}

TEST_CASE("decompose is deterministic and rejects degenerate input", "[decompose]")
{
	REQUIRE(decompose(TriMesh{}, SolverConfig{}).empty());

	const TriMesh cube		 = test::make_box_mesh(1.0F);
	const auto	  first_run	 = decompose(cube, fast_config());
	const auto	  second_run = decompose(cube, fast_config());
	REQUIRE(first_run.size() == second_run.size());
}
