#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <optional>
#include <random>
#include <vector>

#include "pacd/solver/field.hpp"
#include "pacd/solver/primitive_sdf.hpp"
#include "pacd/solver/solver.hpp"

namespace pacd::solver
{

namespace
{

constexpr float ADAM_BETA1	 = 0.9F;
constexpr float ADAM_BETA2	 = 0.999F;
constexpr float ADAM_EPSILON = 1.0e-8F;
constexpr float QUAT_STEP	 = 1.0e-3F; // finite-difference step for quaternion params
constexpr int	PARAM_COUNT	 = 10;		// pos(3) + dims(3) + quat(4)
constexpr int	FIRST_QUAT	 = 6;		// params [6, 10) are the quaternion

enum class Kind : std::uint8_t
{
	SPHERE,
	BOX,
	CYLINDER
};

// pos(0..2), dim0/dim1/dim2 (3..5), quat x/y/z/w (6..9).
using ParamArray = std::array<float, PARAM_COUNT>;

// A Monte-Carlo integration point plus the mesh SDF sampled there.
struct SamplePoint
{
	Vec3  pos;
	float mesh_sdf{0.0F};
};

// Everything the objective needs, bundled to keep helper signatures small.
struct Context
{
	const std::vector<SamplePoint>* samples{nullptr};
	const std::vector<float>*		weight{nullptr}; // 1 = uncovered, 0 = already covered
	float							tau{1.0F};		 // occupancy softness
	float							lambda{1.0F};	 // protrusion penalty weight
};

// Soft occupancy: ~1 well inside (dist << 0), ~0 well outside.
[[nodiscard]] float occupancy(float dist, float tau) noexcept
{
	return 1.0F / (1.0F + std::exp(dist / tau));
}

// A = total soft volume, I = volume inside the mesh, B = fresh (uncovered) interior.
struct Objective
{
	float total{0.0F};
	float interior{0.0F};
	float fresh{0.0F};
};

// Build a Primitive holding the given alternative (assign through the variant base).
template <typename Alt>
[[nodiscard]] Primitive make_primitive(const Alt& alt)
{
	Primitive prim;
	static_cast<PrimitiveBase&>(prim) = alt;
	return prim;
}

[[nodiscard]] Primitive decode(Kind kind, const ParamArray& prm)
{
	const Vec3 pos = vec3(prm.at(0), prm.at(1), prm.at(2));
	if (kind == Kind::SPHERE)
	{
		return make_primitive(Sphere{.pos = pos, .radius = prm.at(3)});
	}
	const Quat rot = normalize(Quat{.x = prm.at(6), .y = prm.at(7), .z = prm.at(8), .w = prm.at(9)});
	if (kind == Kind::BOX)
	{
		return make_primitive(Box{.pos = pos, .size = vec3(prm.at(3), prm.at(4), prm.at(5)), .rot = rot});
	}
	return make_primitive(Cylinder{.pos = pos, .radius = prm.at(3), .height = prm.at(4), .rot = rot});
}

[[nodiscard]] std::array<bool, PARAM_COUNT> active_mask(Kind kind)
{
	std::array<bool, PARAM_COUNT> mask{};
	mask.at(0) = true;
	mask.at(1) = true;
	mask.at(2) = true;
	mask.at(3) = true;
	if (kind == Kind::SPHERE)
	{
		return mask;
	}
	mask.at(4) = true;				  // size.y / height
	mask.at(5) = (kind == Kind::BOX); // size.z (box only)
	mask.at(6) = true;
	mask.at(7) = true;
	mask.at(8) = true;
	mask.at(9) = true;
	return mask;
}

[[nodiscard]] Objective evaluate(const Primitive& prim, const Context& context)
{
	const std::vector<SamplePoint>& samples = *context.samples;
	const std::vector<float>&		weight	= *context.weight;
	Objective						obj;
	for (std::size_t idx = 0; idx < samples.size(); ++idx)
	{
		const float dist_p = sd_primitive(samples.at(idx).pos, prim);
		const float occ_p  = occupancy(dist_p, context.tau);
		const float occ_in = occupancy(std::max(dist_p, samples.at(idx).mesh_sdf), context.tau);
		obj.total += occ_p;
		obj.interior += occ_in;
		obj.fresh += occ_in * weight.at(idx);
	}
	return obj;
}

[[nodiscard]] float loss_at(Kind kind, const ParamArray& prm, const Context& context)
{
	const Objective obj = evaluate(decode(kind, prm), context);
	return -obj.fresh + (context.lambda * (obj.total - obj.interior));
}

void compute_gradient(Kind kind, const ParamArray& prm, const std::array<bool, PARAM_COUNT>& active,
					  const Context& context, float step_len, ParamArray& grad)
{
	for (int prm_i = 0; prm_i < PARAM_COUNT; ++prm_i)
	{
		const auto slot = static_cast<std::size_t>(prm_i);
		if (!active.at(slot))
		{
			grad.at(slot) = 0.0F;
			continue;
		}
		const float step  = (prm_i < FIRST_QUAT) ? step_len : QUAT_STEP;
		ParamArray	plus  = prm;
		ParamArray	minus = prm;
		plus.at(slot) += step;
		minus.at(slot) -= step;
		grad.at(slot) = (loss_at(kind, plus, context) - loss_at(kind, minus, context)) / (2.0F * step);
	}
}

struct AdamState
{
	ParamArray moment1{};
	ParamArray moment2{};
};

void adam_step(const std::array<bool, PARAM_COUNT>& active, const ParamArray& grad, AdamState& adam, int iter,
			   float lr_len, float lr_rot, ParamArray& prm)
{
	const float bias1 = 1.0F - std::pow(ADAM_BETA1, static_cast<float>(iter + 1));
	const float bias2 = 1.0F - std::pow(ADAM_BETA2, static_cast<float>(iter + 1));
	for (int prm_i = 0; prm_i < PARAM_COUNT; ++prm_i)
	{
		const auto slot = static_cast<std::size_t>(prm_i);
		if (!active.at(slot))
		{
			continue;
		}
		const float grd		  = grad.at(slot);
		adam.moment1.at(slot) = (ADAM_BETA1 * adam.moment1.at(slot)) + ((1.0F - ADAM_BETA1) * grd);
		adam.moment2.at(slot) = (ADAM_BETA2 * adam.moment2.at(slot)) + ((1.0F - ADAM_BETA2) * grd * grd);
		const float mhat	  = adam.moment1.at(slot) / bias1;
		const float vhat	  = adam.moment2.at(slot) / bias2;
		const float rate	  = (prm_i < FIRST_QUAT) ? lr_len : lr_rot;
		prm.at(slot) -= rate * mhat / (std::sqrt(vhat) + ADAM_EPSILON);
	}
}

// Clamp dimensions positive and renormalize the quaternion back onto the unit sphere.
void sanitize(ParamArray& prm, float min_dim)
{
	prm.at(3)		= std::max(prm.at(3), min_dim);
	prm.at(4)		= std::max(prm.at(4), min_dim);
	prm.at(5)		= std::max(prm.at(5), min_dim);
	const Quat norm = normalize(Quat{.x = prm.at(6), .y = prm.at(7), .z = prm.at(8), .w = prm.at(9)});
	prm.at(6)		= norm.x;
	prm.at(7)		= norm.y;
	prm.at(8)		= norm.z;
	prm.at(9)		= norm.w;
}

struct Fit
{
	Primitive prim;
	Objective obj;
};

[[nodiscard]] Fit optimize(Kind kind, ParamArray prm, const Context& context, const SolverConfig& config, float scale)
{
	const std::array<bool, PARAM_COUNT> active	 = active_mask(kind);
	const float							step_len = 0.2F * context.tau; // finite-diff step resolves the band
	const float							min_dim	 = 0.02F * scale;
	const float							lr_len	 = config.learning_rate * scale;
	const float							lr_rot	 = config.learning_rate;
	AdamState							adam;
	for (int iter = 0; iter < config.gd_iterations; ++iter)
	{
		ParamArray grad{};
		compute_gradient(kind, prm, active, context, step_len, grad);
		adam_step(active, grad, adam, iter, lr_len, lr_rot, prm);
		sanitize(prm, min_dim);
	}
	const Primitive prim = decode(kind, prm);
	return {.prim = prim, .obj = evaluate(prim, context)};
}

struct Seed
{
	bool  found{false};
	Vec3  pos;
	float clearance{0.0F};
};

// Deepest uncovered interior grid node (largest inscribed clearance).
[[nodiscard]] Seed find_seed(const DistanceField& field, const std::vector<char>& covered)
{
	Seed  seed;
	float deepest = 0.0F;
	for (int ciz = 0; ciz < field.nz; ++ciz)
	{
		for (int ciy = 0; ciy < field.ny; ++ciy)
		{
			for (int cix = 0; cix < field.nx; ++cix)
			{
				const std::size_t node = field.linear_index(cix, ciy, ciz);
				const float		  dist = field.data.at(node);
				if (dist < deepest && covered.at(node) == 0)
				{
					deepest		   = dist;
					seed.found	   = true;
					seed.pos	   = field.node_position(cix, ciy, ciz);
					seed.clearance = -dist;
				}
			}
		}
	}
	return seed;
}

// Local shape frame at the seed, used to warm-start orientation and extents. The
// region is trusted as anisotropic only when it is clearly elongated and there
// are enough interior samples; otherwise callers fall back to an axis-aligned
// isotropic guess (identity orientation).
constexpr float		  FRAME_ANISO_MIN	= 1.25F; // min longest/shortest extent ratio to trust
constexpr float		  FRAME_ANISO_CAP	= 3.0F;	 // clamp on the elongation used for extents
constexpr std::size_t FRAME_MIN_SAMPLES = 24;

struct SeedFrame
{
	// Principal axes (orthonormal, right-handed, descending extent) and the
	// per-axis elongation relative to the shortest axis (>= 1, ratio.z == 1).
	std::array<Vec3, 3> axis{vec3(1.0F, 0.0F, 0.0F), vec3(0.0F, 1.0F, 0.0F), vec3(0.0F, 0.0F, 1.0F)};
	Vec3				ratio{.x = 1.0F, .y = 1.0F, .z = 1.0F};
	bool				anisotropic{false};
};

// Principal-axis analysis of the interior samples around the seed: the covariance
// of the local interior tells us how the region is oriented and elongated, which
// warm-starts each primitive's rotation and extents (see README "seeding").
[[nodiscard]] SeedFrame seed_frame(const std::vector<SamplePoint>& samples, const Seed& seed)
{
	SeedFrame	frame;
	Vec3		mean{};
	std::size_t count = 0;
	for (const SamplePoint& smp : samples)
	{
		if (smp.mesh_sdf < 0.0F)
		{
			mean = mean + (smp.pos - seed.pos);
			++count;
		}
	}
	if (count < FRAME_MIN_SAMPLES)
	{
		return frame; // too little interior to trust -> isotropic fallback
	}

	const Vec3 centroid = seed.pos + (mean * (1.0F / static_cast<float>(count)));
	SymMat3	   cov{};
	for (const SamplePoint& smp : samples)
	{
		if (smp.mesh_sdf >= 0.0F)
		{
			continue;
		}
		const Vec3 off = smp.pos - centroid;
		cov.xx += off.x * off.x;
		cov.yy += off.y * off.y;
		cov.zz += off.z * off.z;
		cov.xy += off.x * off.y;
		cov.xz += off.x * off.z;
		cov.yz += off.y * off.z;
	}

	const SymEigen eigen	  = symmetric_eigen(cov);
	const float	   lambda_max = eigen.values.at(0);
	const float	   lambda_min = std::max(eigen.values.at(2), 1.0e-6F * lambda_max);
	if (lambda_max <= 0.0F || std::sqrt(lambda_max / lambda_min) < FRAME_ANISO_MIN)
	{
		return frame; // essentially isotropic -> isotropic fallback
	}

	const auto elongation = [&eigen, lambda_min](std::size_t idx)
	{
		const float factor = std::sqrt(std::max(eigen.values.at(idx), 0.0F) / lambda_min);
		return std::clamp(factor, 1.0F, FRAME_ANISO_CAP);
	};
	frame.axis.at(0)  = eigen.vectors.at(0);
	frame.axis.at(1)  = eigen.vectors.at(1);
	frame.axis.at(2)  = normalize(cross(eigen.vectors.at(0), eigen.vectors.at(1))); // force right-handed
	frame.ratio		  = vec3(elongation(0), elongation(1), 1.0F);
	frame.anisotropic = true;
	return frame;
}

// Write the oriented quaternion params (slots 6..9) from a local->world rotation.
void set_orientation(ParamArray& prm, const Quat& rot)
{
	prm.at(6) = rot.x;
	prm.at(7) = rot.y;
	prm.at(8) = rot.z;
	prm.at(9) = rot.w;
}

[[nodiscard]] ParamArray seed_params(Kind kind, const Seed& seed, const SeedFrame& frame)
{
	ParamArray	prm{};
	const float clr = seed.clearance;
	prm.at(0)		= seed.pos.x;
	prm.at(1)		= seed.pos.y;
	prm.at(2)		= seed.pos.z;
	prm.at(9)		= 1.0F; // identity quaternion unless a trusted frame overrides it

	if (kind == Kind::SPHERE)
	{
		prm.at(3) = 0.8F * clr; // a sphere ignores orientation; its home is the inscribed ball
		return prm;
	}

	if (kind == Kind::BOX)
	{
		const Vec3 ratio = frame.anisotropic ? frame.ratio : vec3(1.0F, 1.0F, 1.0F);
		prm.at(3)		 = 0.9F * clr * ratio.x;
		prm.at(4)		 = 0.9F * clr * ratio.y;
		prm.at(5)		 = 0.9F * clr * ratio.z;
		if (frame.anisotropic)
		{
			set_orientation(prm, quat_from_basis(frame.axis.at(0), frame.axis.at(1), frame.axis.at(2)));
		}
		return prm;
	}

	// Cylinder: seed its long axis (local +Y) along the dominant principal axis,
	// with the radius left to the constrained minor axes.
	prm.at(3) = 0.6F * clr;
	prm.at(4) = 1.0F * clr * (frame.anisotropic ? frame.ratio.x : 1.0F);
	if (frame.anisotropic)
	{
		const Vec3 axis = frame.axis.at(0);
		const Vec3 side = frame.axis.at(1);
		set_orientation(prm, quat_from_basis(side, axis, normalize(cross(side, axis))));
	}
	return prm;
}

// How far the local sample box extends, as a multiple of the seed clearance.
// Big enough to contain the growing primitive plus a protrusion shell (the mesh
// surface sits at ~1x clearance), small enough that the interior is well sampled
// even for thin parts whose interior is a tiny fraction of the whole bbox.
constexpr float SAMPLE_SPAN = 2.5F;

// Monte-Carlo points in a box around the seed, each carrying the mesh SDF there.
// Local sampling keeps the objective well-conditioned regardless of overall mesh
// size or how thin the part is.
[[nodiscard]] std::vector<SamplePoint> local_samples(const DistanceField& field, const Seed& seed,
													 const SolverConfig& config, std::mt19937& rng)
{
	const float							  extent = SAMPLE_SPAN * seed.clearance;
	std::uniform_real_distribution<float> jitter(-extent, extent);
	std::vector<SamplePoint>			  samples;
	samples.reserve(static_cast<std::size_t>(std::max(config.sample_count, 0)));
	for (int idx = 0; idx < config.sample_count; ++idx)
	{
		const Vec3 pos = seed.pos + vec3(jitter(rng), jitter(rng), jitter(rng));
		samples.push_back({.pos = pos, .mesh_sdf = field.sample(pos)});
	}
	return samples;
}

// Sample weights: 0 where already covered by a placed primitive, else 1.
[[nodiscard]] std::vector<float> compute_weights(const std::vector<SamplePoint>& samples,
												 const std::vector<Primitive>&	 placed)
{
	std::vector<float> weight;
	weight.reserve(samples.size());
	std::ranges::transform(samples, std::back_inserter(weight),
						   [&placed](const SamplePoint& smp)
						   {
							   const Vec3 pos	  = smp.pos;
							   const bool covered = std::ranges::any_of(placed, [pos](const Primitive& prim)
																		{ return sd_primitive(pos, prim) <= 0.0F; });
							   return covered ? 0.0F : 1.0F;
						   });
	return weight;
}

[[nodiscard]] std::size_t count_interior(const DistanceField& field)
{
	return static_cast<std::size_t>(std::ranges::count_if(field.data, [](float dist) { return dist < 0.0F; }));
}

// Mark interior nodes newly covered by `prim`; returns how many were added.
[[nodiscard]] std::size_t mark_covered(const DistanceField& field, const Primitive& prim, std::vector<char>& covered)
{
	std::size_t added = 0;
	for (int ciz = 0; ciz < field.nz; ++ciz)
	{
		for (int ciy = 0; ciy < field.ny; ++ciy)
		{
			for (int cix = 0; cix < field.nx; ++cix)
			{
				const std::size_t node = field.linear_index(cix, ciy, ciz);
				if (field.data.at(node) >= 0.0F || covered.at(node) != 0)
				{
					continue;
				}
				if (sd_primitive(field.node_position(cix, ciy, ciz), prim) <= 0.0F)
				{
					covered.at(node) = 1;
					++added;
				}
			}
		}
	}
	return added;
}

[[nodiscard]] std::vector<Kind> enabled_kinds(const SolverConfig& config)
{
	std::vector<Kind> kinds;
	if (config.use_sphere)
	{
		kinds.push_back(Kind::SPHERE);
	}
	if (config.use_box)
	{
		kinds.push_back(Kind::BOX);
	}
	if (config.use_cylinder)
	{
		kinds.push_back(Kind::CYLINDER);
	}
	return kinds;
}

// Try every enabled primitive type at the seed; keep the best inscribed fit.
[[nodiscard]] std::optional<Fit> best_fit(const Seed& seed, const Context& context, const SolverConfig& config,
										  float scale)
{
	const SeedFrame	   frame = seed_frame(*context.samples, seed);
	std::optional<Fit> best;
	float			   best_fresh = 0.0F;
	for (const Kind kind : enabled_kinds(config))
	{
		const Fit	fit		   = optimize(kind, seed_params(kind, seed, frame), context, config, scale);
		const float protrusion = (fit.obj.total > 0.0F) ? (fit.obj.total - fit.obj.interior) / fit.obj.total : 1.0F;
		if (protrusion > config.max_protrusion)
		{
			continue;
		}
		if (fit.obj.fresh > best_fresh)
		{
			best_fresh = fit.obj.fresh;
			best	   = fit;
		}
	}
	const float min_fresh = config.min_fresh_fraction * static_cast<float>(context.samples->size());
	if (best_fresh < min_fresh)
	{
		return std::nullopt;
	}
	return best;
}

} // namespace

std::vector<Primitive> decompose(const TriMesh& mesh, const SolverConfig& config)
{
	std::vector<Primitive> result;
	if (mesh.triangle_count() == 0 || mesh.vertices.size() < 3)
	{
		return result;
	}

	const DistanceField field		   = build_distance_field(mesh, config.sdf_resolution, config.bbox_padding);
	const std::size_t	total_interior = count_interior(field);
	if (total_interior == 0)
	{
		return result;
	}
	if (config.on_field_built)
	{
		config.on_field_built();
	}

	std::mt19937	  rng(config.seed);
	std::vector<char> covered(field.node_count(), 0);
	std::size_t		  covered_interior = 0;

	while (result.size() < static_cast<std::size_t>(std::max(config.max_primitives, 0)))
	{
		const Seed seed = find_seed(field, covered);
		if (!seed.found)
		{
			break;
		}
		// The seed clearance is the local length scale: it drives the sample box,
		// the occupancy softness, and the optimizer step sizes.
		const float					   scale   = seed.clearance;
		const std::vector<SamplePoint> samples = local_samples(field, seed, config, rng);
		const std::vector<float>	   weight  = compute_weights(samples, result);
		const Context				   context = {.samples = &samples,
												  .weight  = &weight,
												  .tau	   = config.occupancy_tau * scale,
												  .lambda  = config.protrusion_weight};
		const std::optional<Fit>	   best	   = best_fit(seed, context, config, scale);
		if (!best.has_value())
		{
			break;
		}
		result.push_back(best->prim);
		if (config.on_primitive)
		{
			config.on_primitive(result.size(), result.back());
		}
		covered_interior += mark_covered(field, best->prim, covered);
		const float fraction = static_cast<float>(covered_interior) / static_cast<float>(total_interior);
		if (fraction >= config.target_coverage)
		{
			break;
		}
	}
	return result;
}

} // namespace pacd::solver
