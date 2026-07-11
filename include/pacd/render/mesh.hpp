#pragma once

#include <array>
#include <cstdint>
#include <vector>

#include "pacd/render/math.hpp"

namespace pacd::render {

// Axis-aligned bounding box. Default-constructed to an "empty" inverted box so
// that expand() builds it up correctly from the first point.
struct Aabb {
    Vec3 min_point{};
    Vec3 max_point{};
    bool empty{true};

    void expand(Vec3 point) noexcept;
    [[nodiscard]] Vec3 center() const noexcept;
    [[nodiscard]] Vec3 extent() const noexcept;   // max - min
    [[nodiscard]] float radius() const noexcept;   // half-diagonal length
};

using IndexTriangle = std::array<std::uint32_t, 3>;

// Indexed triangle mesh. Normals are per-vertex and may be empty until
// recompute_smooth_normals() runs; the rasterizer can always fall back to a
// geometric face normal for flat shading.
struct Mesh {
    std::vector<Vec3> positions;
    std::vector<Vec3> normals;
    std::vector<IndexTriangle> triangles;

    [[nodiscard]] std::size_t vertex_count() const noexcept { return positions.size(); }
    [[nodiscard]] std::size_t triangle_count() const noexcept { return triangles.size(); }
    [[nodiscard]] bool has_normals() const noexcept { return normals.size() == positions.size(); }

    [[nodiscard]] Aabb bounds() const noexcept;

    // Area-weighted vertex normals. Requires shared (indexed) vertices to yield
    // a smooth result; run weld_vertices() first on triangle-soup input.
    void recompute_smooth_normals();

    // Append a vertex and return its index, for use by primitive builders.
    std::uint32_t add_vertex(Vec3 position);
    void add_triangle(std::uint32_t vertex_a, std::uint32_t vertex_b, std::uint32_t vertex_c);
};

// Merge positions that coincide within `epsilon` (absolute) into shared
// vertices, turning triangle soup (e.g. from STL) into an indexed mesh suitable
// for smooth shading. If `epsilon` is negative it is derived from the bounds.
[[nodiscard]] Mesh weld_vertices(const Mesh& soup, float epsilon = -1.0F);

// Crease-aware ("auto-smooth") normals: face normals are averaged across a
// shared edge only when the angle between the two faces is below
// `crease_degrees`. Curved surfaces (bores, fillets) come out smooth while
// sharp CAD edges stay crisp -- unconditional smoothing melts hard-surface
// parts. Returns triangle soup carrying per-corner normals (positions are
// welded first for connectivity, so `epsilon` matches weld_vertices()).
[[nodiscard]] Mesh with_crease_normals(const Mesh& soup, float crease_degrees = 35.0F,
                                       float epsilon = -1.0F);

}  // namespace pacd::render
