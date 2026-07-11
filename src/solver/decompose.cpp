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

// Inner-loop early stop: quit once the relative loss improvement stays below
// GD_TOLERANCE for GD_PATIENCE consecutive iterations (gd_iterations is the cap).
constexpr float GD_TOLERANCE = 1.0e-3F;
constexpr int	GD_PATIENCE	 = 4;

enum class Kind : std::uint8_t
{
	SPHERE,
	BOX,
	CYLINDER
};

// Optimisable parameters of one primitive. `dims` depends on the kind: sphere ->
// {radius, -, -}; box -> full {size.x, size.y, size.z}; cylinder -> {radius,
// height, -}. The rotation is a unit quaternion but is updated in its so(3)
// tangent space, so there is no over-parameterised quaternion to fight.
struct FitParams
{
	Vec3 pos;
	Vec3 dims;
	Quat rot;
};

// Gradient of the loss w.r.t. FitParams; `rot` is the so(3) (body-frame) tangent
// gradient -- three numbers, not four quaternion components.
struct FitGrad
{
	Vec3 pos;
	Vec3 dims;
	Vec3 rot;
};

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

[[nodiscard]] Primitive decode(Kind kind, const FitParams& prm)
{
	if (kind == Kind::SPHERE)
	{
		return make_primitive(Sphere{.pos = prm.pos, .radius = prm.dims.x});
	}
	if (kind == Kind::BOX)
	{
		return make_primitive(Box{.pos = prm.pos, .size = prm.dims, .rot = prm.rot});
	}
	return make_primitive(Cylinder{.pos = prm.pos, .radius = prm.dims.x, .height = prm.dims.y, .rot = prm.rot});
}

// SDF value at `point` plus the analytic gradient of that value w.r.t. the
// primitive parameters: `pos` in world space, `rot` in the so(3) tangent space,
// `dims` in their own units. For the oriented shapes the gradient is assembled
// from the local-space SDF gradient g via pos = -R g and rot = -(local x g).
struct SampleGrad
{
	float sd{0.0F};
	Vec3  pos;
	Vec3  rot;
	Vec3  dims;
};

[[nodiscard]] SampleGrad sphere_sample_grad(const FitParams& prm, Vec3 point)
{
	const Vec3	rel	   = point - prm.pos;
	const float len	   = length(rel);
	const Vec3	normal = (len > 0.0F) ? (rel * (1.0F / len)) : vec3(1.0F, 0.0F, 0.0F);
	return {.sd = len - prm.dims.x, .pos = -normal, .rot = {}, .dims = vec3(-1.0F, 0.0F, 0.0F)};
}

[[nodiscard]] SampleGrad box_sample_grad(const FitParams& prm, Vec3 point)
{
	const Vec3	local = rotate_inverse(prm.rot, point - prm.pos);
	const Vec3	half  = prm.dims * 0.5F;
	const Vec3	qvec  = vec3(std::abs(local.x) - half.x, std::abs(local.y) - half.y, std::abs(local.z) - half.z);
	const float max_q = std::max({qvec.x, qvec.y, qvec.z});

	Vec3  g_local;
	Vec3  g_dims;
	float dist = 0.0F;
	if (max_q > 0.0F) // outside: dist = |max(q, 0)|; each positive axis contributes
	{
		const Vec3	qpos = vec3(std::max(qvec.x, 0.0F), std::max(qvec.y, 0.0F), std::max(qvec.z, 0.0F));
		const float len	 = length(qpos);
		dist			 = len;
		if (len > 0.0F)
		{
			const Vec3 dir = qpos * (1.0F / len); // d(dist)/d(q)
			g_local		   = vec3(dir.x * std::copysign(1.0F, local.x), dir.y * std::copysign(1.0F, local.y),
								  dir.z * std::copysign(1.0F, local.z));
			g_dims		   = dir * -0.5F;
		}
	}
	else // inside: dist = max_q; the gradient flows through the nearest face only
	{
		dist = max_q;
		if (qvec.x >= qvec.y && qvec.x >= qvec.z)
		{
			g_local = vec3(std::copysign(1.0F, local.x), 0.0F, 0.0F);
			g_dims	= vec3(-0.5F, 0.0F, 0.0F);
		}
		else if (qvec.y >= qvec.z)
		{
			g_local = vec3(0.0F, std::copysign(1.0F, local.y), 0.0F);
			g_dims	= vec3(0.0F, -0.5F, 0.0F);
		}
		else
		{
			g_local = vec3(0.0F, 0.0F, std::copysign(1.0F, local.z));
			g_dims	= vec3(0.0F, 0.0F, -0.5F);
		}
	}
	return {.sd = dist, .pos = -rotate(prm.rot, g_local), .rot = -cross(local, g_local), .dims = g_dims};
}

[[nodiscard]] SampleGrad cylinder_sample_grad(const FitParams& prm, Vec3 point)
{
	const Vec3	local	= rotate_inverse(prm.rot, point - prm.pos);
	const float rho		= std::sqrt((local.x * local.x) + (local.z * local.z));
	const float inv_rho = (rho > 0.0F) ? (1.0F / rho) : 0.0F;
	const float radial	= rho - prm.dims.x;						   // dims.x = radius
	const float axial	= std::abs(local.y) - (prm.dims.y * 0.5F); // dims.y = height

	Vec3  g_local;
	Vec3  g_dims;
	float dist = 0.0F;
	if (radial > 0.0F || axial > 0.0F) // outside
	{
		const float rpos = std::max(radial, 0.0F);
		const float apos = std::max(axial, 0.0F);
		const float len	 = std::sqrt((rpos * rpos) + (apos * apos));
		dist			 = len;
		if (len > 0.0F)
		{
			const float d_radial = rpos / len;
			const float d_axial	 = apos / len;
			g_local				 = vec3(d_radial * local.x * inv_rho, d_axial * std::copysign(1.0F, local.y),
										d_radial * local.z * inv_rho);
			g_dims				 = vec3(-d_radial, -0.5F * d_axial, 0.0F);
		}
	}
	else if (radial >= axial) // inside, the curved wall is nearest
	{
		dist	= radial;
		g_local = vec3(local.x * inv_rho, 0.0F, local.z * inv_rho);
		g_dims	= vec3(-1.0F, 0.0F, 0.0F);
	}
	else // inside, an end cap is nearest
	{
		dist	= axial;
		g_local = vec3(0.0F, std::copysign(1.0F, local.y), 0.0F);
		g_dims	= vec3(0.0F, -0.5F, 0.0F);
	}
	return {.sd = dist, .pos = -rotate(prm.rot, g_local), .rot = -cross(local, g_local), .dims = g_dims};
}

[[nodiscard]] SampleGrad sample_grad(Kind kind, const FitParams& prm, Vec3 point)
{
	if (kind == Kind::SPHERE)
	{
		return sphere_sample_grad(prm, point);
	}
	if (kind == Kind::BOX)
	{
		return box_sample_grad(prm, point);
	}
	return cylinder_sample_grad(prm, point);
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

// Loss and its analytic gradient in a single pass over the samples. The loss is
//   -sum(occ_in * weight) + lambda * (sum(occ_p) - sum(occ_in)),
// matching evaluate(); occ_in only responds to the primitive on the branch where
// it is the active side of max(dist, mesh_sdf).
struct LossGrad
{
	float	loss{0.0F};
	FitGrad grad;
};

[[nodiscard]] LossGrad loss_and_gradient(Kind kind, const FitParams& prm, const Context& context)
{
	const std::vector<SamplePoint>& samples = *context.samples;
	const std::vector<float>&		weight	= *context.weight;
	const float						inv_tau = 1.0F / context.tau;
	LossGrad						out;
	for (std::size_t idx = 0; idx < samples.size(); ++idx)
	{
		const SampleGrad grad_i = sample_grad(kind, prm, samples.at(idx).pos);
		const float		 mesh	= samples.at(idx).mesh_sdf;
		const float		 occ_p	= occupancy(grad_i.sd, context.tau);
		const float		 occ_in = occupancy(std::max(grad_i.sd, mesh), context.tau);
		out.loss += -(occ_in * weight.at(idx)) + (context.lambda * (occ_p - occ_in));

		// d(occupancy)/d(dist) = -inv_tau * occ * (1 - occ); occ_in tracks the
		// primitive only when the primitive is the active side of max(dist, mesh).
		const float d_occ_p	 = -inv_tau * occ_p * (1.0F - occ_p);
		const float d_occ_in = (grad_i.sd > mesh) ? (-inv_tau * occ_in * (1.0F - occ_in)) : 0.0F;
		const float adjoint	 = (-weight.at(idx) * d_occ_in) + (context.lambda * (d_occ_p - d_occ_in));

		out.grad.pos  = out.grad.pos + (grad_i.pos * adjoint);
		out.grad.dims = out.grad.dims + (grad_i.dims * adjoint);
		out.grad.rot  = out.grad.rot + (grad_i.rot * adjoint);
	}
	return out;
}

struct AdamState
{
	FitGrad moment1;
	FitGrad moment2;
};

// One Adam coordinate update for a 3-vector, returning the amount to subtract.
[[nodiscard]] Vec3 adam_delta(Vec3 grad, Vec3& moment1, Vec3& moment2, float bias1, float bias2, float rate)
{
	moment1			= (ADAM_BETA1 * moment1) + ((1.0F - ADAM_BETA1) * grad);
	moment2			= vec3((ADAM_BETA2 * moment2.x) + ((1.0F - ADAM_BETA2) * grad.x * grad.x),
						   (ADAM_BETA2 * moment2.y) + ((1.0F - ADAM_BETA2) * grad.y * grad.y),
						   (ADAM_BETA2 * moment2.z) + ((1.0F - ADAM_BETA2) * grad.z * grad.z));
	const Vec3 mhat = moment1 * (1.0F / bias1);
	const Vec3 vhat = moment2 * (1.0F / bias2);
	return vec3((rate * mhat.x) / (std::sqrt(vhat.x) + ADAM_EPSILON),
				(rate * mhat.y) / (std::sqrt(vhat.y) + ADAM_EPSILON),
				(rate * mhat.z) / (std::sqrt(vhat.z) + ADAM_EPSILON));
}

// Keep the active dimensions positive (which ones are active depends on the kind).
void clamp_dims(Kind kind, FitParams& prm, float min_dim)
{
	prm.dims.x = std::max(prm.dims.x, min_dim); // radius / size.x, always used
	if (kind == Kind::BOX)
	{
		prm.dims.y = std::max(prm.dims.y, min_dim);
		prm.dims.z = std::max(prm.dims.z, min_dim);
	}
	else if (kind == Kind::CYLINDER)
	{
		prm.dims.y = std::max(prm.dims.y, min_dim); // height
	}
}

struct Fit
{
	Primitive prim;
	Objective obj;
};

[[nodiscard]] Fit optimize(Kind kind, FitParams prm, const Context& context, const SolverConfig& config, float scale)
{
	const float min_dim = 0.02F * scale;
	const float lr_len	= config.learning_rate * scale; // world-unit params scale with clearance
	const float lr_rot	= config.learning_rate;			// rotation steps are already dimensionless
	AdamState	adam;
	float		prev_loss = 0.0F;
	int			stalled	  = 0;
	for (int iter = 0; iter < config.gd_iterations; ++iter)
	{
		const LossGrad step	 = loss_and_gradient(kind, prm, context);
		const float	   bias1 = 1.0F - std::pow(ADAM_BETA1, static_cast<float>(iter + 1));
		const float	   bias2 = 1.0F - std::pow(ADAM_BETA2, static_cast<float>(iter + 1));

		prm.pos	 = prm.pos - adam_delta(step.grad.pos, adam.moment1.pos, adam.moment2.pos, bias1, bias2, lr_len);
		prm.dims = prm.dims - adam_delta(step.grad.dims, adam.moment1.dims, adam.moment2.dims, bias1, bias2, lr_len);
		if (kind != Kind::SPHERE)
		{
			// so(3) tangent step composed onto the quaternion: R <- R * exp([theta]).
			const Vec3 theta = -adam_delta(step.grad.rot, adam.moment1.rot, adam.moment2.rot, bias1, bias2, lr_rot);
			prm.rot			 = normalize(prm.rot * quat_from_axis_angle(theta, length(theta)));
		}
		clamp_dims(kind, prm, min_dim);

		if (iter > 0 && std::abs(prev_loss - step.loss) <= GD_TOLERANCE * (std::abs(prev_loss) + ADAM_EPSILON))
		{
			if (++stalled >= GD_PATIENCE)
			{
				break; // converged: the loss has essentially stopped improving
			}
		}
		else
		{
			stalled = 0;
		}
		prev_loss = step.loss;
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

[[nodiscard]] FitParams seed_params(Kind kind, const Seed& seed, const SeedFrame& frame)
{
	const float clr = seed.clearance;
	FitParams	prm{.pos = seed.pos, .dims = {}, .rot = {}}; // identity rotation by default

	if (kind == Kind::SPHERE)
	{
		prm.dims = vec3(0.8F * clr, 0.0F, 0.0F); // a sphere ignores orientation; home is the inscribed ball
		return prm;
	}

	if (kind == Kind::BOX)
	{
		const Vec3 ratio = frame.anisotropic ? frame.ratio : vec3(1.0F, 1.0F, 1.0F);
		prm.dims		 = vec3(0.9F * clr * ratio.x, 0.9F * clr * ratio.y, 0.9F * clr * ratio.z);
		if (frame.anisotropic)
		{
			prm.rot = quat_from_basis(frame.axis.at(0), frame.axis.at(1), frame.axis.at(2));
		}
		return prm;
	}

	// Cylinder: seed its long axis (local +Y) along the dominant principal axis,
	// with the radius left to the constrained minor axes.
	prm.dims = vec3(0.6F * clr, 1.0F * clr * (frame.anisotropic ? frame.ratio.x : 1.0F), 0.0F);
	if (frame.anisotropic)
	{
		const Vec3 axis = frame.axis.at(0);
		const Vec3 side = frame.axis.at(1);
		prm.rot			= quat_from_basis(side, axis, normalize(cross(side, axis)));
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
