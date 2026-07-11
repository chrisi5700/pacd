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

TEST_CASE("quat_from_basis reconstructs a rotation from its axes", "[math]")
{
	// 90 deg about Z: local x -> world +y, local y -> world -x, local z -> world +z.
	const Quat from_basis = quat_from_basis(vec3(0.0F, 1.0F, 0.0F), vec3(-1.0F, 0.0F, 0.0F), vec3(0.0F, 0.0F, 1.0F));
	const Vec3 mapped_x	  = rotate(from_basis, vec3(1.0F, 0.0F, 0.0F));
	REQUIRE(mapped_x.x == Approx(0.0F).margin(EPS));
	REQUIRE(mapped_x.y == Approx(1.0F).margin(EPS));

	// It must agree with the equivalent axis-angle rotation on an arbitrary point.
	const Quat reference = quat_from_axis_angle(vec3(0.0F, 0.0F, 1.0F), PI_F * 0.5F);
	const Vec3 probe	 = vec3(0.3F, -0.7F, 1.1F);
	const Vec3 via_basis = rotate(from_basis, probe);
	const Vec3 via_axis	 = rotate(reference, probe);
	REQUIRE(via_basis.x == Approx(via_axis.x).margin(EPS));
	REQUIRE(via_basis.y == Approx(via_axis.y).margin(EPS));
	REQUIRE(via_basis.z == Approx(via_axis.z).margin(EPS));
}

TEST_CASE("symmetric_eigen orders eigenvalues and finds the dominant axis", "[math]")
{
	// Diagonal matrix: eigenvalues are the diagonal, returned in descending order.
	const SymEigen diagonal = symmetric_eigen(SymMat3{.xx = 1.0F, .yy = 5.0F, .zz = 3.0F});
	REQUIRE(diagonal.values.at(0) == Approx(5.0F).margin(EPS));
	REQUIRE(diagonal.values.at(1) == Approx(3.0F).margin(EPS));
	REQUIRE(diagonal.values.at(2) == Approx(1.0F).margin(EPS));
	REQUIRE(std::abs(diagonal.vectors.at(0).y) == Approx(1.0F).margin(EPS)); // dominant axis is +/-Y

	// A cloud stretched along (1,1,0): cov = 4*(dir dir^T) + I gives eigenvalue 5
	// along dir and 1 on the other two axes.
	const SymEigen tilted = symmetric_eigen(SymMat3{.xx = 3.0F, .yy = 3.0F, .zz = 1.0F, .xy = 2.0F});
	REQUIRE(tilted.values.at(0) == Approx(5.0F).margin(EPS));
	const Vec3 dir = normalize(vec3(1.0F, 1.0F, 0.0F));
	REQUIRE(std::abs(dot(tilted.vectors.at(0), dir)) == Approx(1.0F).margin(1.0e-3F));
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

// World-space long axis of a fitted primitive (its local longest extent rotated
// into world space): +Y for a cylinder, the widest local axis for a box.
[[nodiscard]] Vec3 primitive_long_axis(const Primitive& prim)
{
	if (std::holds_alternative<Cylinder>(prim))
	{
		return rotate(std::get<Cylinder>(prim).rot, vec3(0.0F, 1.0F, 0.0F));
	}
	const Box& box	 = std::get<Box>(prim);
	Vec3	   local = vec3(1.0F, 0.0F, 0.0F);
	if (box.size.y >= box.size.x && box.size.y >= box.size.z)
	{
		local = vec3(0.0F, 1.0F, 0.0F);
	}
	else if (box.size.z >= box.size.x && box.size.z >= box.size.y)
	{
		local = vec3(0.0F, 0.0F, 1.0F);
	}
	return rotate(box.rot, local);
}

[[nodiscard]] Vec3 primitive_center(const Primitive& prim)
{
	if (std::holds_alternative<Sphere>(prim))
	{
		return std::get<Sphere>(prim).pos;
	}
	if (std::holds_alternative<Box>(prim))
	{
		return std::get<Box>(prim).pos;
	}
	return std::get<Cylinder>(prim).pos;
}

// A scalar summary of a primitive's dimensions -- identical for exact replicas.
[[nodiscard]] float primitive_size_signature(const Primitive& prim)
{
	if (std::holds_alternative<Sphere>(prim))
	{
		return std::get<Sphere>(prim).radius;
	}
	if (std::holds_alternative<Box>(prim))
	{
		const Vec3 size = std::get<Box>(prim).size;
		return size.x + size.y + size.z;
	}
	const Cylinder cyl = std::get<Cylinder>(prim);
	return cyl.radius + cyl.height;
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

TEST_CASE("decompose orients a primitive along an oblique beam", "[decompose]")
{
	// A long, thin beam whose axis is rotated 45 deg about Z (an oblique diagonal
	// in the XY plane). Warm-start PCA should orient the fitted primitive's long
	// axis along the beam rather than leaving it axis-aligned -- the whole point of
	// item 1. The beam's world long axis is local +X rotated by `rot`.
	const Quat	  rot  = quat_from_axis_angle(vec3(0.0F, 0.0F, 1.0F), PI_F * 0.25F);
	const TriMesh beam = test::make_beam_mesh(vec3(2.5F, 0.55F, 0.55F), rot);

	SolverConfig config	  = fast_config();
	config.sdf_resolution = 24;
	config.sample_count	  = 1500;
	config.gd_iterations  = 80;
	config.max_primitives = 1; // only the first (deepest-seed) primitive is needed

	const std::vector<Primitive> parts = decompose(beam, config);
	REQUIRE(!parts.empty());
	REQUIRE_FALSE(std::holds_alternative<Sphere>(parts.front())); // a sphere has no long axis

	const Vec3 beam_axis = rotate(rot, vec3(1.0F, 0.0F, 0.0F));
	const Vec3 fit_axis	 = primitive_long_axis(parts.front());
	// Aligned within ~15 deg (|cos| > ~0.87); an axis-aligned fit would score ~0.71.
	REQUIRE(std::abs(dot(beam_axis, fit_axis)) > 0.87F);
}

TEST_CASE("decompose replicates primitives across a mirror symmetry", "[decompose]")
{
	// Two disjoint cubes mirrored across x = 0. Symmetry-aware seeding should fit
	// one lobe and replicate the fit to the other, so every primitive has a mirror
	// partner with *identical* dimensions -- exactness only replication provides,
	// since two independent fits would differ by Monte-Carlo noise.
	const TriMesh mesh	  = test::make_two_box_mesh(0.7F, 1.5F);
	SolverConfig  config  = fast_config();
	config.max_primitives = 12;

	const std::vector<Primitive> parts = decompose(mesh, config);
	REQUIRE(parts.size() >= 2);

	// Both lobes are covered.
	const auto min_x = std::ranges::min(parts, {}, [](const Primitive& prim) { return primitive_center(prim).x; });
	const auto max_x = std::ranges::max(parts, {}, [](const Primitive& prim) { return primitive_center(prim).x; });
	REQUIRE(primitive_center(min_x).x < 0.0F);
	REQUIRE(primitive_center(max_x).x > 0.0F);

	// Every primitive has an exact-size mirror partner in the opposite lobe.
	for (const Primitive& prim : parts)
	{
		const Vec3	center = primitive_center(prim);
		const float sig	   = primitive_size_signature(prim);
		const bool	paired = std::ranges::any_of(parts,
												 [&](const Primitive& other)
												 {
													const Vec3 other_c = primitive_center(other);
													return std::abs(sig - primitive_size_signature(other)) < EPS &&
														   std::abs(other_c.x + center.x) < 0.15F &&
														   std::abs(other_c.y - center.y) < 0.15F &&
														   std::abs(other_c.z - center.z) < 0.15F;
												 });
		REQUIRE(paired);
	}
}

TEST_CASE("decompose is deterministic and rejects degenerate input", "[decompose]")
{
	REQUIRE(decompose(TriMesh{}, SolverConfig{}).empty());

	const TriMesh cube		 = test::make_box_mesh(1.0F);
	const auto	  first_run	 = decompose(cube, fast_config());
	const auto	  second_run = decompose(cube, fast_config());
	REQUIRE(first_run.size() == second_run.size());
}
