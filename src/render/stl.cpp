#include "pacd/render/stl.hpp"

#include <bit>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <fstream>
#include <ios>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include "pacd/render/math.hpp"
#include "pacd/render/mesh.hpp"

namespace pacd::render {

namespace {

using Bytes = std::vector<std::uint8_t>;

constexpr std::size_t BINARY_HEADER = 80;
constexpr std::size_t BINARY_TRIANGLE = 50;

[[nodiscard]] std::expected<Bytes, std::string> read_file(std::string_view path) {
    std::ifstream file(std::string{path}, std::ios::binary | std::ios::ate);
    if (!file) {
        return std::unexpected(std::string{"cannot open file: "} + std::string{path});
    }
    const std::streamoff size = file.tellg();
    if (size <= 0) {
        return std::unexpected(std::string{"empty or unreadable file: "} + std::string{path});
    }
    file.seekg(0, std::ios::beg);
    Bytes buffer(static_cast<std::size_t>(size));
    file.read(reinterpret_cast<char*>(buffer.data()), size);
    if (!file) {
        return std::unexpected(std::string{"short read: "} + std::string{path});
    }
    return buffer;
}

[[nodiscard]] std::uint32_t read_u32(const Bytes& data, std::size_t offset) noexcept {
    return static_cast<std::uint32_t>(data.at(offset)) |
           (static_cast<std::uint32_t>(data.at(offset + 1)) << 8U) |
           (static_cast<std::uint32_t>(data.at(offset + 2)) << 16U) |
           (static_cast<std::uint32_t>(data.at(offset + 3)) << 24U);
}

[[nodiscard]] float read_f32(const Bytes& data, std::size_t offset) noexcept {
    return std::bit_cast<float>(read_u32(data, offset));
}

[[nodiscard]] Vec3 read_vec3(const Bytes& data, std::size_t offset) noexcept {
    return {read_f32(data, offset), read_f32(data, offset + 4), read_f32(data, offset + 8)};
}

void emit_triangle(Mesh& mesh, Vec3 vert0, Vec3 vert1, Vec3 vert2) {
    const Vec3 normal = normalize(cross(vert1 - vert0, vert2 - vert0));
    const std::uint32_t base = mesh.add_vertex(vert0);
    mesh.add_vertex(vert1);
    mesh.add_vertex(vert2);
    mesh.normals.push_back(normal);
    mesh.normals.push_back(normal);
    mesh.normals.push_back(normal);
    mesh.add_triangle(base, base + 1, base + 2);
}

[[nodiscard]] bool looks_binary(const Bytes& data) noexcept {
    if (data.size() < BINARY_HEADER + 4) {
        return false;
    }
    const std::uint32_t count = read_u32(data, BINARY_HEADER);
    const std::size_t expected =
        BINARY_HEADER + 4 + (static_cast<std::size_t>(count) * BINARY_TRIANGLE);
    return data.size() == expected;
}

[[nodiscard]] std::expected<Mesh, std::string> parse_binary(const Bytes& data) {
    const std::uint32_t count = read_u32(data, BINARY_HEADER);
    Mesh mesh;
    mesh.positions.reserve(static_cast<std::size_t>(count) * 3);
    mesh.triangles.reserve(count);
    std::size_t offset = BINARY_HEADER + 4;
    for (std::uint32_t tri = 0; tri < count; ++tri) {
        // Skip the stored face normal (offset..+12); recompute for robustness.
        const Vec3 vert0 = read_vec3(data, offset + 12);
        const Vec3 vert1 = read_vec3(data, offset + 24);
        const Vec3 vert2 = read_vec3(data, offset + 36);
        emit_triangle(mesh, vert0, vert1, vert2);
        offset += BINARY_TRIANGLE;
    }
    return mesh;
}

[[nodiscard]] std::expected<Mesh, std::string> parse_ascii(const Bytes& data) {
    std::string text(data.begin(), data.end());
    std::istringstream stream(text);
    Mesh mesh;
    std::vector<Vec3> pending;
    pending.reserve(3);
    std::string token;
    while (stream >> token) {
        if (token != "vertex") {
            continue;
        }
        Vec3 point{};
        if (!(stream >> point.x >> point.y >> point.z)) {
            return std::unexpected(std::string{"malformed ASCII STL vertex"});
        }
        pending.push_back(point);
        if (pending.size() == 3) {
            emit_triangle(mesh, pending.at(0), pending.at(1), pending.at(2));
            pending.clear();
        }
    }
    if (mesh.triangles.empty()) {
        return std::unexpected(std::string{"no triangles parsed from ASCII STL"});
    }
    return mesh;
}

}  // namespace

std::expected<Mesh, std::string> load_stl(std::string_view path) {
    const std::expected<Bytes, std::string> data = read_file(path);
    if (!data) {
        return std::unexpected(data.error());
    }
    if (looks_binary(*data)) {
        return parse_binary(*data);
    }
    return parse_ascii(*data);
}

}  // namespace pacd::render
