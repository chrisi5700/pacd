#pragma once

#include <vector>

#include "pacd/render/math.hpp"
#include "pacd/render/mesh.hpp"

namespace pacd::render {

// Phong material. `specular_strength` scales the highlight; `shininess` is the
// Phong exponent.
struct Material {
    Vec3 albedo{0.80F, 0.80F, 0.82F};
    float specular_strength{0.30F};
    float shininess{32.0F};
};

// Directional light defined by the direction the light travels (world space).
struct DirectionalLight {
    Vec3 direction{-0.4F, -1.0F, -0.6F};
    Vec3 color{1.0F, 1.0F, 1.0F};
    float intensity{1.0F};
};

// A drawable: CPU geometry plus its placement, material and draw flags. The
// viewer owns the matching GPU buffers separately, so a Scene stays pure data
// and can be built/inspected without an OpenGL context.
struct Object {
    Mesh mesh{};
    Mat4 model{Mat4::identity()};
    Material material{};
    bool visible{true};
    bool wireframe{false};
};

struct Scene {
    std::vector<Object> objects;
    std::vector<DirectionalLight> lights;
    Vec3 ambient{0.14F, 0.14F, 0.16F};
    Vec3 background{0.09F, 0.10F, 0.12F};

    // Union of every visible object's bounds, transformed into world space.
    [[nodiscard]] Aabb bounds() const;
};

// A reasonable two-light studio setup (key + fill) used by the default scene.
[[nodiscard]] std::vector<DirectionalLight> default_lights();

}  // namespace pacd::render
