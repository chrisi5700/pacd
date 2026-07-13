#include "pacd/render/camera.hpp"

#include <algorithm>
#include <cmath>

#include "pacd/render/math.hpp"
#include "pacd/render/mesh.hpp"

namespace pacd::render
{

namespace
{
constexpr float PITCH_LIMIT = 1.55334F; // ~89 degrees, avoids look-at gimbal flip
constexpr Vec3	WORLD_UP{0.0F, 1.0F, 0.0F};
} // namespace

Vec3 OrbitCamera::eye() const noexcept
{
	const Vec3 direction{std::cos(m_pitch) * std::sin(m_yaw), std::sin(m_pitch), std::cos(m_pitch) * std::cos(m_yaw)};
	return m_target + (direction * m_distance);
}

void OrbitCamera::orbit(float yaw_delta, float pitch_delta) noexcept
{
	m_yaw += yaw_delta;
	m_pitch = std::clamp(m_pitch + pitch_delta, -PITCH_LIMIT, PITCH_LIMIT);
}

void OrbitCamera::pan(float delta_x, float delta_y, float viewport_height) noexcept
{
	const float height	= std::max(viewport_height, 1.0F);
	const Vec3	forward = normalize(m_target - eye());
	const Vec3	right	= normalize(cross(forward, WORLD_UP));
	const Vec3	up_axis = cross(right, forward);
	// World units per pixel at the target plane.
	const float speed = m_distance * 2.0F * std::tan(radians(m_fov_y_degrees) * 0.5F) / height;
	m_target		  = m_target + (right * (-delta_x * speed)) + (up_axis * (delta_y * speed));
}

void OrbitCamera::dolly(float amount) noexcept
{
	const float factor = std::exp(-amount * 0.15F);
	m_distance		   = std::clamp(m_distance * factor, 1e-5F, 1e9F);
}

void OrbitCamera::frame(const Aabb& box) noexcept
{
	const float radius	 = std::max(box.radius(), 1e-5F);
	m_target			 = box.center();
	const float half_fov = radians(m_fov_y_degrees) * 0.5F;
	m_distance			 = (radius / std::sin(half_fov)) * 1.25F;
	m_z_near			 = std::max(m_distance * 0.001F, 1e-4F);
	m_z_far				 = (m_distance + radius) * 4.0F;
}

Mat4 OrbitCamera::view() const noexcept
{
	return look_at(eye(), m_target, WORLD_UP);
}

Mat4 OrbitCamera::projection(float aspect) const noexcept
{
	const float safe_aspect = std::max(aspect, 1e-3F);
	return perspective(radians(m_fov_y_degrees), safe_aspect, m_z_near, m_z_far);
}

} // namespace pacd::render
