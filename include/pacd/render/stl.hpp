#pragma once

#include <expected>
#include <string>
#include <string_view>

#include "pacd/render/mesh.hpp"

namespace pacd::render {

// Load an STL file, auto-detecting ASCII vs binary encoding. The result is
// triangle soup (per-triangle vertices) with geometric face normals; run
// weld_vertices() on it to obtain shared vertices and smooth normals.
//
// Returns an error string (never throws) on I/O failure or malformed data, in
// keeping with the project's std::expected error-handling convention.
[[nodiscard]] std::expected<Mesh, std::string> load_stl(std::string_view path);

}  // namespace pacd::render
