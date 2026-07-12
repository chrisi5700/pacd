//
// pacd solver API: approximate a triangle mesh with a small set of convex
// primitives (spheres / boxes / cylinders) whose union is inscribed in the mesh,
// for use as a physics collision proxy. See README "Approach" for the greedy
// outer loop / gradient-descent inner loop design.
//

#pragma once

#include <cstddef>
#include <functional>
#include <vector>

#include "pacd/solver/mesh.hpp"
#include "pacd/solver/primitives.hpp"

namespace pacd::solver
{

// Tunable knobs for a single decomposition run. Defaults are provisional
// starting points, not tuned values.
struct SolverConfig
{
	// Termination -- stop at whichever trips first.
	float target_coverage{0.95F}; // fraction of interior volume to fill, in [0, 1]
	int	  max_primitives{32};	  // hard cap on the number of primitives placed

	// Volume sampling: the mesh SDF and interior coverage are evaluated on a grid
	// whose longest axis spans this many cells.
	int sdf_resolution{64};

	// Inner loop: gradient descent inflating one inscribed primitive. Step sizes
	// scale with the local seed clearance, so learning_rate is dimensionless.
	int	  gd_iterations{150};
	float learning_rate{0.03F};

	// Softness. `occupancy_tau` is the sigmoid width used to turn an SDF into a
	// differentiable inside/outside weight, as a fraction of the local seed
	// clearance (smaller = crisper). `smooth_min_k` blends the primitive union so
	// gradient reaches every nearby primitive, not just the closest one.
	float occupancy_tau{0.2F};
	float smooth_min_k{0.05F};

	// Inscribed-constraint strength: penalty weight on primitive volume that
	// protrudes outside the mesh (lambda in the "balloon" objective).
	float protrusion_weight{4.0F};

	// Monte-Carlo integration points for the inner-loop objective, and the RNG
	// seed that places them (fixed for determinism).
	int		 sample_count{4096};
	unsigned seed{1337U};

	// Bounding-box padding (fraction of the diagonal) for the distance field, so
	// it carries an exterior shell for the protrusion term.
	float bbox_padding{0.1F};

	// A fitted primitive is rejected if more than this fraction of its volume
	// pokes outside the mesh (keeps the union inscribed).
	float max_protrusion{0.08F};

	// Stop placing once the best candidate captures less than this fraction of
	// the sample budget as fresh interior (diminishing returns).
	float min_fresh_fraction{0.004F};

	// Primitive vocabulary tried at each greedy step (try-all-keep-best).
	bool use_sphere{true};
	bool use_box{true};
	bool use_cylinder{true};

	// Detect the mesh's global symmetry (mirror / n-fold about the principal axes)
	// and replicate each placed primitive across the symmetry group, so symmetric
	// regions are filled from a single fit. Replicas are validated independently,
	// so an approximate symmetry never forces a protruding or redundant primitive.
	bool use_symmetry{true};

	// After placement, greedily replace adjacent primitive pairs with a single
	// re-fitted primitive (a few gradient-descent steps to inscribe the joint
	// region). A merge is accepted only if one primitive re-covers at least
	// `merge_retain` of the pair's interior (counting what surviving primitives
	// still hold) without protruding, so it trades primitive count for a bounded
	// coverage give-back. Raise `merge_retain` toward 1 for near-lossless (fewer,
	// cleaner merges); lower it for a more parsimonious, slightly looser proxy.
	bool  merge_primitives{true};
	float merge_retain{0.97F};

	// Optional progress hooks (empty by default), for long-running decompositions.
	// `on_field_built` fires once after the mesh SDF grid is built; `on_primitive`
	// fires after each primitive is placed, with the running count.
	std::function<void()>									  on_field_built;
	std::function<void(std::size_t placed, const Primitive&)> on_primitive;
};

// Decompose `mesh` into an inscribed union of convex primitives approximating
// its interior. Primitives are placed greedily until `target_coverage` of the
// interior volume is filled or `max_primitives` is reached.
[[nodiscard]] std::vector<Primitive> decompose(const TriMesh& mesh, const SolverConfig& config = {});

} // namespace pacd::solver
