//
// score_composites -- decompose every mesh in a directory of composite test
// fixtures and report how well the fitted primitives approximate each solid.
//
// It is a measurement/regression harness, not part of the solver: it exists so a
// change to the fitter can be judged on real multi-part shapes (dumbbell, jenga
// tower, wheel axle, ...) instead of by eye. For each `<name>.stl` it decomposes
// the mesh with the viewer's live settings and scores the union of primitives on
// an independent, finer distance-field grid:
//
//   coverage = interior filled by the union        (want high -- "did it grow?")
//   spill    = union volume outside the mesh        (want low  -- "inscribed?")
//   IoU      = intersection / union of proxy & mesh (the single headline number)
//
// The optional `<name>.json` sidecar (the generator's ground truth) supplies the
// intended part count and type mix, shown next to what we produced so parsimony
// regressions (four spheres where one cylinder belongs) are obvious.
//

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iterator>
#include <numeric>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

#include <fmt/core.h>

#include "pacd/render/stl.hpp"
#include "pacd/solver/field.hpp"
#include "pacd/solver/primitive_sdf.hpp"
#include "pacd/solver/render_bridge.hpp"
#include "pacd/solver/solver.hpp"

namespace
{

namespace fs = std::filesystem;
using pacd::solver::Box;
using pacd::solver::Cylinder;
using pacd::solver::DistanceField;
using pacd::solver::Primitive;
using pacd::solver::Sphere;
using pacd::solver::Vec3;

// Grid resolution for the *scoring* field. Deliberately finer than the solver's
// own grid so the metric measures the shape, not the fitter's sampling.
constexpr int	EVAL_RESOLUTION = 72;
constexpr float EVAL_PADDING	= 0.1F;

// Solver knobs mirroring the interactive viewer, so the harness scores exactly
// what a user sees on screen.
[[nodiscard]] pacd::solver::SolverConfig harness_config()
{
	pacd::solver::SolverConfig config;
	config.sdf_resolution = 32;
	config.sample_count	  = 1200;
	config.gd_iterations  = 60;
	config.max_primitives = 24;
	return config;
}

// Node counts from one pass over the scoring grid.
struct Metrics
{
	std::size_t interior{0}; // grid nodes inside the mesh
	std::size_t covered{0};	 // interior nodes also inside the union (intersection)
	std::size_t spill{0};	 // union nodes outside the mesh (proxy \ mesh)
};

[[nodiscard]] Metrics score(const DistanceField& field, const std::vector<Primitive>& parts)
{
	Metrics out;
	for (int ciz = 0; ciz < field.nz; ++ciz)
	{
		for (int ciy = 0; ciy < field.ny; ++ciy)
		{
			for (int cix = 0; cix < field.nx; ++cix)
			{
				const Vec3 pos		   = field.node_position(cix, ciy, ciz);
				const bool inside_mesh = field.at_index(cix, ciy, ciz) < 0.0F;
				const bool inside_union =
					std::ranges::any_of(parts, [pos](const Primitive& prim) { return sd_primitive(pos, prim) <= 0.0F; });
				out.interior += inside_mesh ? 1 : 0;
				out.covered += (inside_mesh && inside_union) ? 1 : 0;
				out.spill += (!inside_mesh && inside_union) ? 1 : 0;
			}
		}
	}
	return out;
}

// Primitive type histogram, ours or the ground truth's.
struct Mix
{
	int box{0};
	int sphere{0};
	int cyl{0};
};

[[nodiscard]] Mix mix_of(const std::vector<Primitive>& parts)
{
	Mix mix;
	for (const Primitive& prim : parts)
	{
		mix.box += std::holds_alternative<Box>(prim) ? 1 : 0;
		mix.sphere += std::holds_alternative<Sphere>(prim) ? 1 : 0;
		mix.cyl += std::holds_alternative<Cylinder>(prim) ? 1 : 0;
	}
	return mix;
}

[[nodiscard]] std::string read_file(const fs::path& path)
{
	std::ifstream stream(path, std::ios::binary);
	return {std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>()};
}

[[nodiscard]] int count_occurrences(std::string_view text, std::string_view needle)
{
	int count = 0;
	for (std::size_t pos = text.find(needle); pos != std::string_view::npos; pos = text.find(needle, pos + needle.size()))
	{
		++count;
	}
	return count;
}

// Intended part count and type mix from the generator's JSON sidecar (absent ->
// all zeros, printed as "--"). A deliberately dependency-free scan: the sidecars
// are flat and we only need these counts, not a parse tree.
struct Expected
{
	int	 parts{0};
	Mix	 mix;
	bool present{false};
};

[[nodiscard]] Expected read_expected(const fs::path& json_path)
{
	if (!fs::exists(json_path))
	{
		return {};
	}
	const std::string text = read_file(json_path);
	return {.parts = count_occurrences(text, "\"kind\""),
			.mix   = {.box	  = count_occurrences(text, "\"box\""),
					  .sphere = count_occurrences(text, "\"sphere\""),
					  .cyl	  = count_occurrences(text, "\"cylinder\"")},
			.present = true};
}

struct Row
{
	std::string name;
	Mix			mix;
	Expected	expected;
	Metrics		metrics;
	double		millis{0.0};
};

[[nodiscard]] float percent(std::size_t part, std::size_t whole)
{
	return (whole > 0) ? (100.0F * static_cast<float>(part) / static_cast<float>(whole)) : 0.0F;
}

// coverage / spill / IoU, all in [0, 1] * 100 where shown as percents.
[[nodiscard]] float coverage_pct(const Metrics& met) { return percent(met.covered, met.interior); }
[[nodiscard]] float spill_pct(const Metrics& met) { return percent(met.spill, met.covered + met.spill); }
[[nodiscard]] float iou(const Metrics& met) { return percent(met.covered, met.interior + met.spill) / 100.0F; }

[[nodiscard]] std::optional<Row> process(const fs::path& stl_path)
{
	const std::expected<pacd::render::Mesh, std::string> loaded = pacd::render::load_stl(stl_path.string());
	if (!loaded.has_value())
	{
		fmt::print(stderr, "  skip {}: {}\n", stl_path.filename().string(), loaded.error());
		return std::nullopt;
	}
	const pacd::solver::TriMesh tri	  = pacd::solver::to_tri_mesh(*loaded);
	const DistanceField			field = build_distance_field(tri, EVAL_RESOLUTION, EVAL_PADDING);

	const auto							 start = std::chrono::steady_clock::now();
	const std::vector<Primitive>		 parts = pacd::solver::decompose(tri, harness_config());
	const std::chrono::duration<double>	 dur   = std::chrono::steady_clock::now() - start;

	fs::path json_path = stl_path;
	json_path.replace_extension(".json");
	return Row{.name	 = stl_path.stem().string(),
			   .mix		 = mix_of(parts),
			   .expected = read_expected(json_path),
			   .metrics	 = score(field, parts),
			   .millis	 = dur.count() * 1000.0};
}

void print_row(const Row& row)
{
	const std::string got_mix = fmt::format("B{}/S{}/C{}", row.mix.box, row.mix.sphere, row.mix.cyl);
	const std::string exp_mix =
		row.expected.present
			? fmt::format("B{}/S{}/C{}", row.expected.mix.box, row.expected.mix.sphere, row.expected.mix.cyl)
			: std::string("--");
	const std::string prims =
		row.expected.present
			? fmt::format("{}/{}", row.mix.box + row.mix.sphere + row.mix.cyl, row.expected.parts)
			: fmt::format("{}", row.mix.box + row.mix.sphere + row.mix.cyl);
	fmt::print("{:<16} prims {:>5}  mix {:<9} (exp {:<9})  cov {:5.1f}%  spill {:4.1f}%  IoU {:.3f}  {:6.0f}ms\n",
			   row.name, prims, got_mix, exp_mix, coverage_pct(row.metrics), spill_pct(row.metrics), iou(row.metrics),
			   row.millis);
}

// Mean of a metric over the rows (unweighted per mesh -- each shape counts once).
[[nodiscard]] float mean(const std::vector<Row>& rows, float (*metric)(const Metrics&))
{
	if (rows.empty())
	{
		return 0.0F;
	}
	const float sum = std::transform_reduce(rows.begin(), rows.end(), 0.0F, std::plus<>{},
											[metric](const Row& row) { return metric(row.metrics); });
	return sum / static_cast<float>(rows.size());
}

[[nodiscard]] std::vector<fs::path> stl_files(const fs::path& dir)
{
	std::vector<fs::path> files;
	if (!fs::is_directory(dir))
	{
		return files;
	}
	for (const fs::directory_entry& entry : fs::directory_iterator(dir))
	{
		if (entry.is_regular_file() && entry.path().extension() == ".stl")
		{
			files.push_back(entry.path());
		}
	}
	std::ranges::sort(files);
	return files;
}

} // namespace

int main(int argc, char** argv)
{
	const std::span<char* const>  raw(argv, static_cast<std::size_t>(argc));
	std::vector<std::string_view> tokens;
	std::ranges::transform(raw.subspan(raw.empty() ? 0 : 1), std::back_inserter(tokens),
						   [](char* const arg) { return std::string_view{arg}; });
	const fs::path dir = tokens.empty() ? fs::path("resources/composite") : fs::path(tokens.at(0));

	const std::vector<fs::path> files = stl_files(dir);
	if (files.empty())
	{
		fmt::print(stderr, "no .stl files under {}\n", dir.string());
		return 1;
	}

	fmt::print("scoring {} composite mesh(es) in {}\n\n", files.size(), dir.string());
	std::vector<Row> rows;
	for (const fs::path& file : files)
	{
		if (const std::optional<Row> row = process(file))
		{
			print_row(*row);
			rows.push_back(*row);
		}
	}

	fmt::print("\nmean over {} mesh(es):  cov {:.1f}%   spill {:.1f}%   IoU {:.3f}\n", rows.size(),
			   mean(rows, coverage_pct), mean(rows, spill_pct), mean(rows, iou));
	return 0;
}
