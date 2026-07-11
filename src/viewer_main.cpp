// Command-line front-end for the pacd interactive renderer.
//
//   pacd-viewer [options] [mesh.stl ...]
//
// With no mesh arguments it shows the built-in primitive gallery. Meshes are
// overlaid (each in a palette colour), which is the layout used to compare a
// source mesh with its convex-decomposition pieces.

#include <algorithm>
#include <cstdlib>
#include <expected>
#include <iterator>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <fmt/core.h>

#include "pacd/render/scene.hpp"
#include "pacd/render/scenes.hpp"
#include "pacd/render/viewer.hpp"

// The GPU driver (NVIDIA/Mesa) allocates internal buffers it never frees before
// process exit; under AddressSanitizer these surface as leaks originating in the
// driver .so, not our code. Suppress them at the source so the sanitized build
// is usable interactively without setting LSAN/ASAN env vars.
#ifdef __SANITIZE_ADDRESS__
// NOLINTNEXTLINE(bugprone-reserved-identifier,readability-identifier-naming,cert-dcl37-c,cert-dcl51-cpp)
extern "C" const char* __lsan_default_suppressions() {
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
extern "C" const char* __lsan_default_options() { return "print_suppressions=0"; }
#endif

namespace {

struct Args {
    int width{1280};
    int height{800};
    bool hidden{false};
    bool wire{false};
    bool flat{false};
    bool help{false};
    int frames{-1};
    std::string screenshot;
    std::vector<std::string> files;
};

void print_usage() {
    fmt::print(
        "pacd-viewer [options] [mesh.stl ...]\n"
        "\n"
        "  No mesh -> primitive gallery. Multiple meshes are overlaid in\n"
        "  distinct colours (source vs. convex-hull pieces).\n"
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
        "          F frame, W wireframe, S screenshot, Esc quit.\n");
}

[[nodiscard]] int to_int(std::string_view text, int fallback) {
    std::istringstream stream{std::string{text}};
    int value = fallback;
    if (stream >> value) {
        return value;
    }
    return fallback;
}

[[nodiscard]] Args parse_args(const std::vector<std::string_view>& tokens) {
    Args args;
    for (std::size_t index = 0; index < tokens.size(); ++index) {
        const std::string_view token = tokens.at(index);
        const bool has_next = (index + 1) < tokens.size();
        if (token == "-h" || token == "--help") {
            args.help = true;
        } else if (token == "--wire") {
            args.wire = true;
        } else if (token == "--flat") {
            args.flat = true;
        } else if (token == "--hidden") {
            args.hidden = true;
        } else if (token == "--width" && has_next) {
            args.width = to_int(tokens.at(++index), args.width);
        } else if (token == "--height" && has_next) {
            args.height = to_int(tokens.at(++index), args.height);
        } else if (token == "--frames" && has_next) {
            args.frames = to_int(tokens.at(++index), args.frames);
        } else if (token == "--screenshot" && has_next) {
            args.screenshot = std::string{tokens.at(++index)};
        } else if (!token.empty() && token.front() != '-') {
            args.files.emplace_back(token);
        } else {
            fmt::print(stderr, "warning: ignoring unknown option '{}'\n", token);
        }
    }
    return args;
}

[[nodiscard]] bool build_scene(const Args& args, pacd::render::Scene& out) {
    if (args.files.empty()) {
        out = pacd::render::make_primitive_gallery();
        return true;
    }
    std::expected<pacd::render::Scene, std::string> loaded =
        pacd::render::load_stl_scene(args.files, !args.flat);
    if (!loaded) {
        fmt::print(stderr, "error loading meshes: {}\n", loaded.error());
        return false;
    }
    out = std::move(*loaded);
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    const std::span<char* const> raw(argv, static_cast<std::size_t>(argc));
    std::vector<std::string_view> tokens;
    tokens.reserve(raw.size());
    if (!raw.empty()) {
        std::ranges::transform(raw.subspan(1), std::back_inserter(tokens),
                               [](char* const arg) { return std::string_view{arg}; });
    }

    const Args args = parse_args(tokens);
    if (args.help) {
        print_usage();
        return EXIT_SUCCESS;
    }

    pacd::render::Scene scene;
    if (!build_scene(args, scene)) {
        return EXIT_FAILURE;
    }

    pacd::render::ViewerOptions options;
    options.width = args.width;
    options.height = args.height;
    options.visible = !args.hidden;
    options.title = "pacd viewer";

    pacd::render::Viewer viewer(std::move(options));
    if (!viewer.valid()) {
        fmt::print(stderr, "renderer initialisation failed: {}\n", viewer.error());
        return EXIT_FAILURE;
    }
    viewer.set_scene(std::move(scene));
    viewer.set_wireframe(args.wire);

    const bool offscreen_task = args.frames > 0 || !args.screenshot.empty();
    if (offscreen_task) {
        const int frames = args.frames > 0 ? args.frames : 3;
        viewer.run_frames(frames);
        if (!args.screenshot.empty()) {
            if (!viewer.save_screenshot(args.screenshot)) {
                fmt::print(stderr, "failed to save screenshot to '{}'\n", args.screenshot);
                return EXIT_FAILURE;
            }
            fmt::print("saved screenshot: {}\n", args.screenshot);
        }
        return EXIT_SUCCESS;
    }

    viewer.run();
    return EXIT_SUCCESS;
}
