#pragma once

#include "pacd/render/math.hpp"
#include "pacd/render/mesh.hpp"

namespace pacd::render {

// Turntable / arcball-style inspection camera. It always looks at a target
// point; the mouse orbits around it, pans the target in the view plane, and the
// scroll wheel dollies in and out. This is the standard "inspect a part" camera
// found in CAD/DCC tools.
class OrbitCamera {
   public:
    void orbit(float yaw_delta, float pitch_delta) noexcept;
    void pan(float delta_x, float delta_y, float viewport_height) noexcept;
    void dolly(float amount) noexcept;

    // Fit the camera so the whole box is comfortably in view and pick sensible
    // near/far planes for the box's scale (STL parts range over many orders of
    // magnitude, so depth planes cannot be fixed constants).
    void frame(const Aabb& box) noexcept;

    [[nodiscard]] Vec3 eye() const noexcept;
    [[nodiscard]] Vec3 target() const noexcept { return m_target; }
    [[nodiscard]] Mat4 view() const noexcept;
    [[nodiscard]] Mat4 projection(float aspect) const noexcept;

    void set_fov_y_degrees(float degrees) noexcept { m_fov_y_degrees = degrees; }
    [[nodiscard]] float fov_y_degrees() const noexcept { return m_fov_y_degrees; }

   private:
    Vec3 m_target{0.0F, 0.0F, 0.0F};
    float m_distance{3.0F};
    float m_yaw{radians(45.0F)};    // azimuth around +Y
    float m_pitch{radians(20.0F)};  // elevation
    float m_z_near{0.01F};
    float m_z_far{1000.0F};
    float m_fov_y_degrees{45.0F};
};

}  // namespace pacd::render
