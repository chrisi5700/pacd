// Command-line front-end for the pacd viewer.
//
//   pacd-viewer [options] [mesh.stl ...]
//
// With mesh arguments, each file is loaded, decomposed into convex primitives,
// and shown as solid coloured primitives overlaid on the original mesh (drawn as
// a wireframe cage). Left/Right arrows cycle through the files; T toggles the
// original cage so you can compare it against the primitive approximation. With
// no mesh arguments it shows the built-in primitive gallery.

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <expected>
#include <fmt/core.h>
#include <imgui.h>
#include <iterator>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

#include "pacd/render/math.hpp"
#include "pacd/render/mesh.hpp"
#include "pacd/render/primitives.hpp"
#include "pacd/render/scene.hpp"
#include "pacd/render/scenes.hpp"
#include "pacd/render/stl.hpp"
#include "pacd/render/viewer.hpp"
#include "pacd/solver/primitives.hpp"
#include "pacd/solver/render_bridge.hpp"
#include "pacd/solver/solver.hpp"

// The GPU driver (NVIDIA/Mesa) allocates internal buffers it never frees before
// process exit; under AddressSanitizer these surface as leaks in the driver .so.
// Suppress them so the sanitized build is usable without setting env vars.
#ifdef __SANITIZE_ADDRESS__
// NOLINTNEXTLINE(bugprone-reserved-identifier,readability-identifier-naming,cert-dcl37-c,cert-dcl51-cpp)
extern "C" const char* __lsan_default_suppressions()
{
	return "leak:libnvidia-glcore\n"
		   "leak:libnvidia-eglcore\n"
		   "leak:libGLX\n"
		   "leak:libGLdispatch\n"
		   "leak:libEGL\n"
		   "leak:swrast\n"
		   "leak:libglapi\n"
		   "leak:i965_dri\n";
}
// NOLINTNEXTLINE(bugprone-reserved-identifier,readability-identifier-naming,cert-dcl37-c,cert-dcl51-cpp)
extern "C" const char* __lsan_default_options()
{
	return "print_suppressions=0";
}
#endif

using namespace pacd::render;
namespace solver = pacd::solver;

namespace
{

struct Args
{
	int						 width{1280};
	int						 height{800};
	bool					 hidden{false};
	bool					 wire{false};
	bool					 flat{false};
	bool					 help{false};
	int						 frames{-1};
	std::string				 screenshot;
	std::vector<std::string> files;
};

void print_usage()
{
	fmt::print("pacd-viewer [options] [mesh.stl ...]\n"
			   "\n"
			   "  Each mesh is decomposed into convex primitives, shown solid and\n"
			   "  coloured, overlaid on the original mesh (a wireframe cage).\n"
			   "  No mesh -> primitive gallery.\n"
			   "\n"
			   "Options:\n"
			   "  --width N / --height N   window size (default 1280x800)\n"
			   "  --wire                   start in wireframe\n"
			   "  --flat                   keep raw facet normals (no smoothing)\n"
			   "  --hidden                 offscreen window (no visible window)\n"
			   "  --frames N               render N frames then exit\n"
			   "  --screenshot FILE        save a PNG after rendering, then exit\n"
			   "  -h, --help               this message\n"
			   "\n"
			   "Controls: left-drag orbit, middle/right-drag pan, scroll zoom,\n"
			   "          Left/Right cycle files, T toggle original mesh,\n"
			   "          F frame, W wireframe, S screenshot, Esc quit.\n");
}

[[nodiscard]] int to_int(std::string_view text, int fallback)
{
	std::istringstream stream{std::string{text}};
	int				   value = fallback;
	if (stream >> value)
	{
		return value;
	}
	return fallback;
}

[[nodiscard]] Args parse_args(const std::vector<std::string_view>& tokens)
{
	Args args;
	for (std::size_t index = 0; index < tokens.size(); ++index)
	{
		const std::string_view token	= tokens.at(index);
		const bool			   has_next = (index + 1) < tokens.size();
		if (token == "-h" || token == "--help")
		{
			args.help = true;
		}
		else if (token == "--wire")
		{
			args.wire = true;
		}
		else if (token == "--flat")
		{
			args.flat = true;
		}
		else if (token == "--hidden")
		{
			args.hidden = true;
		}
		else if (token == "--width" && has_next)
		{
			args.width = to_int(tokens.at(++index), args.width);
		}
		else if (token == "--height" && has_next)
		{
			args.height = to_int(tokens.at(++index), args.height);
		}
		else if (token == "--frames" && has_next)
		{
			args.frames = to_int(tokens.at(++index), args.frames);
		}
		else if (token == "--screenshot" && has_next)
		{
			args.screenshot = std::string{tokens.at(++index)};
		}
		else if (!token.empty() && token.front() != '-')
		{
			args.files.emplace_back(token);
		}
		else
		{
			fmt::print(stderr, "warning: ignoring unknown option '{}'\n", token);
		}
	}
	return args;
}

[[nodiscard]] const char* kind_name(const solver::Primitive& prim)
{
	if (std::holds_alternative<solver::Sphere>(prim))
	{
		return "sphere";
	}
	if (std::holds_alternative<solver::Box>(prim))
	{
		return "box";
	}
	return "cylinder";
}

// Rotation matrix (column-major) from a unit quaternion.
[[nodiscard]] Mat4 quat_to_mat4(const solver::Quat& rot)
{
	Mat4 mat = Mat4::identity();
	mat.set(0, 0, 1.0F - (2.0F * ((rot.y * rot.y) + (rot.z * rot.z))));
	mat.set(0, 1, 2.0F * ((rot.x * rot.y) - (rot.w * rot.z)));
	mat.set(0, 2, 2.0F * ((rot.x * rot.z) + (rot.w * rot.y)));
	mat.set(1, 0, 2.0F * ((rot.x * rot.y) + (rot.w * rot.z)));
	mat.set(1, 1, 1.0F - (2.0F * ((rot.x * rot.x) + (rot.z * rot.z))));
	mat.set(1, 2, 2.0F * ((rot.y * rot.z) - (rot.w * rot.x)));
	mat.set(2, 0, 2.0F * ((rot.x * rot.z) - (rot.w * rot.y)));
	mat.set(2, 1, 2.0F * ((rot.y * rot.z) + (rot.w * rot.x)));
	mat.set(2, 2, 1.0F - (2.0F * ((rot.x * rot.x) + (rot.y * rot.y))));
	return mat;
}

// A drawable Object for one fitted primitive: analytic mesh + placement matrix.
[[nodiscard]] Object primitive_object(const solver::Primitive& prim, Vec3 color)
{
	Object object;
	if (std::holds_alternative<solver::Sphere>(prim))
	{
		const auto& sphere = std::get<solver::Sphere>(prim);
		object.mesh		   = make_uv_sphere(sphere.radius);
		object.model	   = translation(Vec3{sphere.pos.x, sphere.pos.y, sphere.pos.z});
	}
	else if (std::holds_alternative<solver::Box>(prim))
	{
		const auto& box = std::get<solver::Box>(prim);
		object.mesh		= make_box(Vec3{box.size.x, box.size.y, box.size.z});
		object.model	= translation(Vec3{box.pos.x, box.pos.y, box.pos.z}) * quat_to_mat4(box.rot);
	}
	else
	{
		const auto& cyl = std::get<solver::Cylinder>(prim);
		object.mesh		= make_cylinder(cyl.radius, cyl.height);
		object.model	= translation(Vec3{cyl.pos.x, cyl.pos.y, cyl.pos.z}) * quat_to_mat4(cyl.rot);
	}
	object.material.albedo = color;
	return object;
}

// Original mesh as a wireframe cage (object 0, toggled by T) plus one solid,
// palette-coloured Object per fitted primitive.
[[nodiscard]] Scene overlay_scene(Mesh original, const std::vector<solver::Primitive>& parts)
{
	Scene scene;
	scene.lights = default_lights();

	Object cage;
	cage.mesh			 = std::move(original);
	cage.material.albedo = Vec3{0.85F, 0.85F, 0.88F};
	cage.wireframe		 = true;
	scene.objects.push_back(std::move(cage));

	const std::vector<Vec3>& colors = palette();
	for (std::size_t index = 0; index < parts.size(); ++index)
	{
		scene.objects.push_back(primitive_object(parts.at(index), colors.at(index % colors.size())));
	}
	return scene;
}

[[nodiscard]] solver::SolverConfig viewer_config()
{
	solver::SolverConfig config;
	config.sdf_resolution = 32;
	config.sample_count	  = 1200;
	config.gd_iterations  = 60;
	config.max_primitives = 24;
	config.on_field_built = [] { fmt::print(stderr, "  distance field built; fitting primitives...\n"); };
	config.on_primitive	  = [](std::size_t placed, const solver::Primitive& prim)
	{ fmt::print(stderr, "  placed primitive {} ({})\n", placed, kind_name(prim)); };
	return config;
}

[[nodiscard]] std::vector<std::string_view> collect_tokens(int argc, char** argv)
{
	const std::span<char* const>  raw(argv, static_cast<std::size_t>(argc));
	std::vector<std::string_view> tokens;
	tokens.reserve(raw.size());
	if (!raw.empty())
	{
		std::ranges::transform(raw.subspan(1), std::back_inserter(tokens),
							   [](char* const arg) { return std::string_view{arg}; });
	}
	return tokens;
}

[[nodiscard]] ViewerOptions make_options(const Args& args)
{
	ViewerOptions options;
	options.width	= args.width;
	options.height	= args.height;
	options.visible = !args.hidden;
	options.title	= "pacd viewer";
	return options;
}

// Outcome of one decomposition, surfaced in the panel's status line.
struct RunResult
{
	std::size_t count{0};
	long long	ms{0};
	bool		ok{false};
};

// Set by the ImGui panel (which runs mid-frame) and drained by the app loop, so
// the blocking work it asks for happens outside the ImGui frame.
struct PanelRequest
{
	bool redecompose{false};
	int	 cycle{0}; // -1 previous file, +1 next file, 0 none
};

// --- ImGui config panel --------------------------------------------------
// Each section mutates `config` in place; the app re-runs decompose on request.

void budget_controls(solver::SolverConfig& config)
{
	ImGui::SeparatorText("Budget");
	ImGui::SliderInt("max primitives", &config.max_primitives, 1, 64);
	ImGui::SliderFloat("target coverage", &config.target_coverage, 0.0F, 1.0F);
	ImGui::Checkbox("sphere", &config.use_sphere);
	ImGui::SameLine();
	ImGui::Checkbox("box", &config.use_box);
	ImGui::SameLine();
	ImGui::Checkbox("cylinder", &config.use_cylinder);
	ImGui::Checkbox("symmetry (replicate fits across detected symmetry)", &config.use_symmetry);
}

void quality_controls(solver::SolverConfig& config)
{
	ImGui::SeparatorText("Quality / cost");
	ImGui::SliderInt("SDF resolution", &config.sdf_resolution, 8, 96);
	ImGui::SliderInt("samples", &config.sample_count, 200, 8000);
	ImGui::SliderInt("GD iterations", &config.gd_iterations, 10, 300);
	ImGui::SliderFloat("learning rate", &config.learning_rate, 0.001F, 0.2F, "%.3f");
}

void advanced_controls(solver::SolverConfig& config)
{
	if (!ImGui::CollapsingHeader("Advanced"))
	{
		return;
	}
	ImGui::SliderFloat("occupancy tau", &config.occupancy_tau, 0.01F, 1.0F);
	ImGui::SliderFloat("smooth-min k", &config.smooth_min_k, 0.0F, 0.5F);
	ImGui::SliderFloat("protrusion weight", &config.protrusion_weight, 0.0F, 20.0F);
	ImGui::SliderFloat("max protrusion", &config.max_protrusion, 0.0F, 0.5F);
	ImGui::SliderFloat("min fresh frac", &config.min_fresh_fraction, 0.0F, 0.05F, "%.4f");
}

// Build the panel. Widgets edit `config` live; the buttons/keys only *record*
// intent into `request` (TextUnformatted avoids ImGui's variadic Text overloads,
// which the strict build's cppcoreguidelines-pro-type-vararg check rejects).
void config_panel(solver::SolverConfig& config, std::string_view file, std::size_t which, std::size_t total,
				  const RunResult& last, PanelRequest& request)
{
	ImGui::SetNextWindowSize(ImVec2(400.0F, 0.0F), ImGuiCond_FirstUseEver);
	ImGui::Begin("pacd - decomposition");

	ImGui::TextUnformatted(fmt::format("[{}/{}] {}", which + 1, total, file).c_str());
	const std::string status = last.ok ? fmt::format("{} primitives  -  {} ms", last.count, last.ms) : "no run yet";
	ImGui::TextUnformatted(status.c_str());

	budget_controls(config);
	quality_controls(config);
	advanced_controls(config);

	ImGui::Separator();
	if (ImGui::Button("Re-decompose (R)"))
	{
		request.redecompose = true;
	}
	ImGui::SameLine();
	if (ImGui::Button("< Prev"))
	{
		request.cycle = -1;
	}
	ImGui::SameLine();
	if (ImGui::Button("Next >"))
	{
		request.cycle = 1;
	}
	ImGui::TextUnformatted("re-decompose blocks; progress -> stderr");
	ImGui::End();
}

// Load a mesh, decompose it, and show the overlay (cage + primitives). Logs
// progress because decomposition can take a while.
[[nodiscard]] RunResult load_into_viewer(Viewer& viewer, const std::string& path, bool flat,
										 const solver::SolverConfig& config, std::size_t which, std::size_t total)
{
	fmt::print(stderr, "[{}/{}] loading {}\n", which + 1, total, path);
	const std::expected<Mesh, std::string> loaded = load_stl(path);
	if (!loaded)
	{
		fmt::print(stderr, "  load failed: {}\n", loaded.error());
		return {};
	}
	Mesh display = flat ? *loaded : with_crease_normals(*loaded);
	viewer.set_scene(overlay_scene(display, {})); // show the cage while we think
	viewer.draw_once();

	const solver::TriMesh tri = solver::to_tri_mesh(*loaded);
	fmt::print(stderr, "  decomposing ({} triangles)...\n", tri.triangle_count());
	const auto							 start = std::chrono::steady_clock::now();
	const std::vector<solver::Primitive> parts = solver::decompose(tri, config);
	const auto took = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start);
	fmt::print(stderr, "  {} primitives in {} ms\n", parts.size(), took.count());
	const RunResult result{.count = parts.size(), .ms = static_cast<long long>(took.count()), .ok = true};
	viewer.set_scene(overlay_scene(std::move(display), parts));
	return result;
}

// Interactive session over a list of mesh files. The ImGui panel edits `config`
// live; arrow keys / Prev-Next buttons cycle files, R (or the button) re-runs the
// decomposition with the current config. Requests the panel records mid-frame are
// drained here, after the frame, so the blocking decompose never opens a nested
// ImGui frame.
void run_files(Viewer& viewer, const Args& args, solver::SolverConfig& config)
{
	std::size_t	 index = 0;
	RunResult	 last_run;
	PanelRequest request;

	auto load_index = [&](std::size_t which)
	{
		index	 = which;
		last_run = load_into_viewer(viewer, args.files.at(which), args.flat, config, which, args.files.size());
	};
	auto advance = [&](int direction)
	{
		const int count = static_cast<int>(args.files.size());
		int		  next	= (static_cast<int>(index) + direction) % count;
		if (next < 0)
		{
			next += count;
		}
		load_index(static_cast<std::size_t>(next));
	};

	load_index(0);

	viewer.set_on_gui([&] { config_panel(config, args.files.at(index), index, args.files.size(), last_run, request); });
	viewer.set_on_cycle(advance);
	viewer.set_on_redecompose([&] { load_index(index); });
	viewer.set_on_toggle_original(
		[&viewer]
		{
			if (!viewer.scene().objects.empty())
			{
				Object& cage = viewer.scene().objects.front();
				cage.visible = !cage.visible;
			}
		});

	while (viewer.pump())
	{
		// Drain requests the panel recorded this frame, now that the ImGui frame
		// is closed and blocking work is safe.
		if (request.cycle != 0)
		{
			advance(request.cycle);
		}
		else if (request.redecompose)
		{
			load_index(index);
		}
		request = {};
	}
}

// Render offscreen (N frames / a screenshot) then report; returns the exit code.
[[nodiscard]] int run_offscreen(Viewer& viewer, const Args& args)
{
	const int frames = args.frames > 0 ? args.frames : 3;
	viewer.run_frames(frames);
	if (!args.screenshot.empty())
	{
		if (!viewer.save_screenshot(args.screenshot))
		{
			fmt::print(stderr, "failed to save screenshot to '{}'\n", args.screenshot);
			return EXIT_FAILURE;
		}
		fmt::print("saved screenshot: {}\n", args.screenshot);
	}
	return EXIT_SUCCESS;
}

} // namespace

int main(int argc, char** argv)
{
	const Args args = parse_args(collect_tokens(argc, argv));
	if (args.help)
	{
		print_usage();
		return EXIT_SUCCESS;
	}

	Viewer viewer(make_options(args));
	if (!viewer.valid())
	{
		fmt::print(stderr, "renderer initialisation failed: {}\n", viewer.error());
		return EXIT_FAILURE;
	}
	viewer.set_wireframe(args.wire);

	solver::SolverConfig config	   = viewer_config();
	const bool			 has_files = !args.files.empty();

	// Offscreen runs (screenshot / fixed frame count) decompose once and never
	// show the interactive panel.
	if (args.frames > 0 || !args.screenshot.empty())
	{
		if (has_files)
		{
			std::ignore = load_into_viewer(viewer, args.files.at(0), args.flat, config, 0, args.files.size());
		}
		else
		{
			viewer.set_scene(make_primitive_gallery());
		}
		return run_offscreen(viewer, args);
	}

	if (has_files)
	{
		run_files(viewer, args, config);
	}
	else
	{
		viewer.set_scene(make_primitive_gallery());
		viewer.run();
	}
	return EXIT_SUCCESS;
}
