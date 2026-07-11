#include "pacd/render/scene.hpp"

#include <array>
#include <vector>

#include "pacd/render/math.hpp"
#include "pacd/render/mesh.hpp"

namespace pacd::render {

Aabb Scene::bounds() const {
    Aabb world{};
    for (const Object& object : objects) {
        if (!object.visible) {
            continue;
        }
        const Aabb local = object.mesh.bounds();
        if (local.empty) {
            continue;
        }
        // Transform all eight corners of the local box into world space.
        const std::array<Vec3, 2> extremes{local.min_point, local.max_point};
        for (int corner = 0; corner < 8; ++corner) {
            const Vec3 point{extremes.at(static_cast<std::size_t>(corner & 1)).x,
                             extremes.at(static_cast<std::size_t>((corner >> 1) & 1)).y,
                             extremes.at(static_cast<std::size_t>((corner >> 2) & 1)).z};
            const Vec4 transformed = transform(object.model, point);
            world.expand({transformed.x, transformed.y, transformed.z});
        }
    }
    return world;
}

std::vector<DirectionalLight> default_lights() {
    return {
        DirectionalLight{.direction = normalize(Vec3{-0.5F, -0.9F, -0.4F}),
                         .color = {1.0F, 0.98F, 0.94F},
                         .intensity = 1.0F},
        DirectionalLight{.direction = normalize(Vec3{0.6F, -0.2F, 0.5F}),
                         .color = {0.35F, 0.40F, 0.50F},
                         .intensity = 1.0F},
    };
}

}  // namespace pacd::render
