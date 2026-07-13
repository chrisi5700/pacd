#include "pacd/render/mesh.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <map>
#include <vector>

#include "pacd/render/math.hpp"

namespace pacd::render {

void Aabb::expand(Vec3 point) noexcept {
    if (empty) {
        min_point = point;
        max_point = point;
        empty = false;
        return;
    }
    min_point = min(min_point, point);
    max_point = max(max_point, point);
}

Vec3 Aabb::center() const noexcept { return (min_point + max_point) * 0.5F; }

Vec3 Aabb::extent() const noexcept { return max_point - min_point; }

float Aabb::radius() const noexcept { return length(extent()) * 0.5F; }

Aabb Mesh::bounds() const noexcept {
    Aabb box{};
    for (const Vec3 position : positions) {
        box.expand(position);
    }
    return box;
}

void Mesh::recompute_smooth_normals() {
    normals.assign(positions.size(), Vec3{0.0F, 0.0F, 0.0F});
    for (const IndexTriangle& tri : triangles) {
        const auto i0 = static_cast<std::size_t>(tri.at(0));
        const auto i1 = static_cast<std::size_t>(tri.at(1));
        const auto i2 = static_cast<std::size_t>(tri.at(2));
        if (i0 >= positions.size() || i1 >= positions.size() || i2 >= positions.size()) {
            continue;
        }
        // Cross product magnitude is proportional to triangle area, giving an
        // area-weighted accumulation "for free".
        const Vec3 face =
            cross(positions.at(i1) - positions.at(i0), positions.at(i2) - positions.at(i0));
        normals.at(i0) = normals.at(i0) + face;
        normals.at(i1) = normals.at(i1) + face;
        normals.at(i2) = normals.at(i2) + face;
    }
    std::ranges::transform(normals, normals.begin(), [](Vec3 value) { return normalize(value); });
}

std::uint32_t Mesh::add_vertex(Vec3 position) {
    const auto index = static_cast<std::uint32_t>(positions.size());
    positions.push_back(position);
    return index;
}

void Mesh::add_triangle(std::uint32_t vertex_a, std::uint32_t vertex_b, std::uint32_t vertex_c) {
    triangles.push_back(IndexTriangle{vertex_a, vertex_b, vertex_c});
}

namespace {

// Welded connectivity: unique positions plus triangles indexing into them.
struct WeldResult {
    std::vector<Vec3> positions;
    std::vector<IndexTriangle> triangles;
};

[[nodiscard]] float resolve_epsilon(const Mesh& soup, float epsilon) {
    if (epsilon >= 0.0F) {
        return epsilon;
    }
    const float span = soup.bounds().radius();
    return std::max(1e-8F, 1e-5F * std::max(span, 1.0F));
}

[[nodiscard]] WeldResult weld_indexed(const Mesh& soup, float epsilon) {
    const float inv_eps = 1.0F / epsilon;
    using Key = std::array<std::int64_t, 3>;
    const auto quantize = [inv_eps](Vec3 point) -> Key {
        return {std::llround(point.x * inv_eps), std::llround(point.y * inv_eps),
                std::llround(point.z * inv_eps)};
    };

    WeldResult out;
    std::map<Key, std::uint32_t> lookup;
    const auto intern = [&](Vec3 point) -> std::uint32_t {
        const Key key = quantize(point);
        const auto found = lookup.find(key);
        if (found != lookup.end()) {
            return found->second;
        }
        const auto index = static_cast<std::uint32_t>(out.positions.size());
        out.positions.push_back(point);
        lookup.emplace(key, index);
        return index;
    };

    out.triangles.reserve(soup.triangles.size());
    for (const IndexTriangle& tri : soup.triangles) {
        const auto i0 = static_cast<std::size_t>(tri.at(0));
        const auto i1 = static_cast<std::size_t>(tri.at(1));
        const auto i2 = static_cast<std::size_t>(tri.at(2));
        if (i0 >= soup.positions.size() || i1 >= soup.positions.size() ||
            i2 >= soup.positions.size()) {
            continue;
        }
        out.triangles.push_back(IndexTriangle{intern(soup.positions.at(i0)),
                                              intern(soup.positions.at(i1)),
                                              intern(soup.positions.at(i2))});
    }
    return out;
}

// Per-face unit normal, area, and the faces incident to each welded vertex.
struct FaceData {
    std::vector<Vec3> normals;
    std::vector<float> areas;
    std::vector<std::vector<std::uint32_t>> vertex_faces;
};

[[nodiscard]] FaceData build_face_data(const WeldResult& welded) {
    const std::size_t face_count = welded.triangles.size();
    FaceData data;
    data.normals.assign(face_count, Vec3{0.0F, 0.0F, 0.0F});
    data.areas.assign(face_count, 0.0F);
    data.vertex_faces.assign(welded.positions.size(), {});
    for (std::size_t face = 0; face < face_count; ++face) {
        const IndexTriangle& tri = welded.triangles.at(face);
        const Vec3 pos0 = welded.positions.at(tri.at(0));
        const Vec3 pos1 = welded.positions.at(tri.at(1));
        const Vec3 pos2 = welded.positions.at(tri.at(2));
        const Vec3 raw = cross(pos1 - pos0, pos2 - pos0);
        data.areas.at(face) = 0.5F * length(raw);
        data.normals.at(face) = normalize(raw);
        for (std::size_t k = 0; k < 3; ++k) {
            data.vertex_faces.at(tri.at(k)).push_back(static_cast<std::uint32_t>(face));
        }
    }
    return data;
}

}  // namespace

Mesh weld_vertices(const Mesh& soup, float epsilon) {
    const WeldResult welded = weld_indexed(soup, resolve_epsilon(soup, epsilon));
    Mesh mesh;
    mesh.positions = welded.positions;
    mesh.triangles = welded.triangles;
    mesh.recompute_smooth_normals();
    return mesh;
}

Mesh with_crease_normals(const Mesh& soup, float crease_degrees, float epsilon) {
    const WeldResult welded = weld_indexed(soup, resolve_epsilon(soup, epsilon));
    const FaceData faces = build_face_data(welded);
    const float cos_threshold = std::cos(radians(crease_degrees));

    Mesh out;
    const std::size_t face_count = welded.triangles.size();
    out.positions.reserve(face_count * 3);
    out.normals.reserve(face_count * 3);
    out.triangles.reserve(face_count);
    for (std::size_t face = 0; face < face_count; ++face) {
        const IndexTriangle& tri = welded.triangles.at(face);
        const Vec3 face_normal = faces.normals.at(face);
        const auto base = static_cast<std::uint32_t>(out.positions.size());
        for (std::size_t k = 0; k < 3; ++k) {
            const auto shared = static_cast<std::size_t>(tri.at(k));
            // Average only the incident faces within the crease angle so hard
            // CAD edges stay sharp while curved surfaces stay smooth.
            Vec3 acc{0.0F, 0.0F, 0.0F};
            for (const std::uint32_t neighbor : faces.vertex_faces.at(shared)) {
                if (dot(face_normal, faces.normals.at(neighbor)) >= cos_threshold) {
                    acc = acc + (faces.normals.at(neighbor) * faces.areas.at(neighbor));
                }
            }
            const Vec3 normal = (length(acc) > 0.0F) ? normalize(acc) : face_normal;
            out.positions.push_back(welded.positions.at(shared));
            out.normals.push_back(normal);
        }
        out.add_triangle(base, base + 1, base + 2);
    }
    return out;
}

}  // namespace pacd::render
