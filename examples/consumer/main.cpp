//
// Minimal downstream consumer of the pacd solver library. Builds a unit cube and
// decomposes it into inscribed convex primitives -- exercising the public API
// surface that `find_package(pacd)` exposes through the `pacd::solver` target.
//
// This doubles as the CI install smoke test: it must configure, compile as C++20,
// link against the installed static library, and run without the viewer/GL/logging
// stack anywhere in sight.
//

#include <cstddef>
#include <cstdio>
#include <vector>

#include <pacd/solver/mesh.hpp>
#include <pacd/solver/solver.hpp>

int main()
{
	using namespace pacd::solver;

	// Unit cube [0,1]^3 with consistent outward-facing winding (the solver signs the
	// SDF with a winding number, so orientation must be consistent).
	TriMesh mesh;
	mesh.vertices = {
		vec3(0.0F, 0.0F, 0.0F), vec3(1.0F, 0.0F, 0.0F), vec3(1.0F, 1.0F, 0.0F), vec3(0.0F, 1.0F, 0.0F),
		vec3(0.0F, 0.0F, 1.0F), vec3(1.0F, 0.0F, 1.0F), vec3(1.0F, 1.0F, 1.0F), vec3(0.0F, 1.0F, 1.0F),
	};
	mesh.triangles = {
		{0, 3, 2}, {0, 2, 1}, // z = 0
		{4, 5, 6}, {4, 6, 7}, // z = 1
		{0, 1, 5}, {0, 5, 4}, // y = 0
		{3, 7, 6}, {3, 6, 2}, // y = 1
		{0, 4, 7}, {0, 7, 3}, // x = 0
		{1, 2, 6}, {1, 6, 5}, // x = 1
	};

	SolverConfig config;
	config.max_primitives = 4;
	config.sdf_resolution = 24; // keep the smoke test quick
	config.gd_iterations  = 40;

	const std::vector<Primitive> parts = decompose(mesh, config);

	std::printf("pacd: decomposed a cube into %zu primitive(s)\n", parts.size());
	for (std::size_t i = 0; i < parts.size(); ++i)
	{
		std::printf("  [%zu] %s\n", i, parts[i].kind_name());
	}

	// A cube is the easiest possible input -- an empty result means the library is
	// mislinked or broken, so fail the smoke test.
	return parts.empty() ? 1 : 0;
}
