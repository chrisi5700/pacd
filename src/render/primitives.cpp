#include "pacd/render/primitives.hpp"

#include <array>
#include <cmath>
#include <cstdint>

#include "pacd/render/math.hpp"
#include "pacd/render/mesh.hpp"

namespace pacd::render {

namespace {

// Append four coplanar corners (counter-clockwise around `normal`) as two
// triangles sharing the given normal.
void add_quad(Mesh& mesh, const std::array<Vec3, 4>& corners, Vec3 normal) {
    const auto base = static_cast<std::uint32_t>(mesh.positions.size());
    for (const Vec3 corner : corners) {
        mesh.positions.push_back(corner);
        mesh.normals.push_back(normal);
    }
    mesh.add_triangle(base + 0, base + 1, base + 2);
    mesh.add_triangle(base + 0, base + 2, base + 3);
}

std::uint32_t push_vertex(Mesh& mesh, Vec3 position, Vec3 normal) {
    const auto index = static_cast<std::uint32_t>(mesh.positions.size());
    mesh.positions.push_back(position);
    mesh.normals.push_back(normal);
    return index;
}

}  // namespace

Mesh make_box(Vec3 size) {
    const Vec3 half = size * 0.5F;
    const float hx = half.x;
    const float hy = half.y;
    const float hz = half.z;
    Mesh mesh;
    add_quad(mesh, {Vec3{hx, -hy, -hz}, {hx, hy, -hz}, {hx, hy, hz}, {hx, -hy, hz}},
             {1.0F, 0.0F, 0.0F});
    add_quad(mesh, {Vec3{-hx, -hy, hz}, {-hx, hy, hz}, {-hx, hy, -hz}, {-hx, -hy, -hz}},
             {-1.0F, 0.0F, 0.0F});
    add_quad(mesh, {Vec3{-hx, hy, hz}, {hx, hy, hz}, {hx, hy, -hz}, {-hx, hy, -hz}},
             {0.0F, 1.0F, 0.0F});
    add_quad(mesh, {Vec3{-hx, -hy, -hz}, {hx, -hy, -hz}, {hx, -hy, hz}, {-hx, -hy, hz}},
             {0.0F, -1.0F, 0.0F});
    add_quad(mesh, {Vec3{-hx, -hy, hz}, {hx, -hy, hz}, {hx, hy, hz}, {-hx, hy, hz}},
             {0.0F, 0.0F, 1.0F});
    add_quad(mesh, {Vec3{hx, -hy, -hz}, {-hx, -hy, -hz}, {-hx, hy, -hz}, {hx, hy, -hz}},
             {0.0F, 0.0F, -1.0F});
    return mesh;
}

Mesh make_plane(float width, float depth) {
    const float half_w = width * 0.5F;
    const float half_d = depth * 0.5F;
    Mesh mesh;
    add_quad(mesh,
             {Vec3{-half_w, 0.0F, half_d}, {half_w, 0.0F, half_d}, {half_w, 0.0F, -half_d},
              {-half_w, 0.0F, -half_d}},
             {0.0F, 1.0F, 0.0F});
    return mesh;
}

Mesh make_uv_sphere(float radius, int segments, int rings) {
    const int seg_count = std::max(segments, 3);
    const int ring_count = std::max(rings, 2);
    Mesh mesh;
    for (int ring = 0; ring <= ring_count; ++ring) {
        const float phi = PI_F * (static_cast<float>(ring) / static_cast<float>(ring_count));
        const float cos_phi = std::cos(phi);
        const float sin_phi = std::sin(phi);
        for (int seg = 0; seg <= seg_count; ++seg) {
            const float theta =
                (2.0F * PI_F) * (static_cast<float>(seg) / static_cast<float>(seg_count));
            const Vec3 normal{sin_phi * std::cos(theta), cos_phi, sin_phi * std::sin(theta)};
            push_vertex(mesh, normal * radius, normal);
        }
    }
    const auto stride = static_cast<std::uint32_t>(seg_count + 1);
    const auto ring_span = static_cast<std::uint32_t>(ring_count);
    const auto seg_span = static_cast<std::uint32_t>(seg_count);
    for (std::uint32_t ring = 0; ring < ring_span; ++ring) {
        for (std::uint32_t seg = 0; seg < seg_span; ++seg) {
            const std::uint32_t top = (ring * stride) + seg;
            const std::uint32_t bottom = top + stride;
            mesh.add_triangle(top, bottom, top + 1);
            mesh.add_triangle(top + 1, bottom, bottom + 1);
        }
    }
    return mesh;
}

Mesh make_cylinder(float radius, float height, int segments) {
    const int seg_count = std::max(segments, 3);
    const float half_h = height * 0.5F;
    Mesh mesh;
    const std::uint32_t top_center = push_vertex(mesh, {0.0F, half_h, 0.0F}, {0.0F, 1.0F, 0.0F});
    const std::uint32_t bot_center = push_vertex(mesh, {0.0F, -half_h, 0.0F}, {0.0F, -1.0F, 0.0F});
    for (int seg = 0; seg < seg_count; ++seg) {
        const float theta0 =
            (2.0F * PI_F) * (static_cast<float>(seg) / static_cast<float>(seg_count));
        const float theta1 =
            (2.0F * PI_F) * (static_cast<float>(seg + 1) / static_cast<float>(seg_count));
        const Vec3 dir0{std::cos(theta0), 0.0F, std::sin(theta0)};
        const Vec3 dir1{std::cos(theta1), 0.0F, std::sin(theta1)};
        const Vec3 top0{dir0.x * radius, half_h, dir0.z * radius};
        const Vec3 top1{dir1.x * radius, half_h, dir1.z * radius};
        const Vec3 bot0{dir0.x * radius, -half_h, dir0.z * radius};
        const Vec3 bot1{dir1.x * radius, -half_h, dir1.z * radius};
        // Side (radial normals).
        add_quad(mesh, {top0, top1, bot1, bot0}, normalize(dir0 + dir1));
        // Caps (fans to the pole centres).
        const std::uint32_t top_a = push_vertex(mesh, top0, {0.0F, 1.0F, 0.0F});
        const std::uint32_t top_b = push_vertex(mesh, top1, {0.0F, 1.0F, 0.0F});
        mesh.add_triangle(top_center, top_a, top_b);
        const std::uint32_t bot_a = push_vertex(mesh, bot0, {0.0F, -1.0F, 0.0F});
        const std::uint32_t bot_b = push_vertex(mesh, bot1, {0.0F, -1.0F, 0.0F});
        mesh.add_triangle(bot_center, bot_b, bot_a);
    }
    return mesh;
}

Mesh make_cone(float radius, float height, int segments) {
    const int seg_count = std::max(segments, 3);
    const float half_h = height * 0.5F;
    Mesh mesh;
    const std::uint32_t bot_center = push_vertex(mesh, {0.0F, -half_h, 0.0F}, {0.0F, -1.0F, 0.0F});
    for (int seg = 0; seg < seg_count; ++seg) {
        const float theta0 =
            (2.0F * PI_F) * (static_cast<float>(seg) / static_cast<float>(seg_count));
        const float theta1 =
            (2.0F * PI_F) * (static_cast<float>(seg + 1) / static_cast<float>(seg_count));
        const Vec3 dir0{std::cos(theta0), 0.0F, std::sin(theta0)};
        const Vec3 dir1{std::cos(theta1), 0.0F, std::sin(theta1)};
        const Vec3 base0{dir0.x * radius, -half_h, dir0.z * radius};
        const Vec3 base1{dir1.x * radius, -half_h, dir1.z * radius};
        const Vec3 apex{0.0F, half_h, 0.0F};
        // Side normal tilts by the slope: radial * height + up * radius.
        const Vec3 side_normal =
            normalize(Vec3{(dir0.x + dir1.x) * 0.5F * height, radius, (dir0.z + dir1.z) * 0.5F * height});
        const std::uint32_t side_a = push_vertex(mesh, base0, side_normal);
        const std::uint32_t side_b = push_vertex(mesh, base1, side_normal);
        const std::uint32_t apex_v = push_vertex(mesh, apex, side_normal);
        mesh.add_triangle(side_a, side_b, apex_v);
        // Base cap.
        const std::uint32_t cap_a = push_vertex(mesh, base0, {0.0F, -1.0F, 0.0F});
        const std::uint32_t cap_b = push_vertex(mesh, base1, {0.0F, -1.0F, 0.0F});
        mesh.add_triangle(bot_center, cap_b, cap_a);
    }
    return mesh;
}

}  // namespace pacd::render
