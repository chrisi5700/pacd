#pragma once

// Minimal float-only linear algebra for the software renderer.
//
// Kept dependency-free (no glm) so the render library builds cleanly under the
// strict `llm-vcpkg` preset without pulling third-party headers through
// clang-tidy. Matrices are column-major (OpenGL convention) so the same code
// could later back an OpenGL viewer without transposing.

#include <array>
#include <cmath>
#include <cstddef>

namespace pacd::render {

inline constexpr float PI_F = 3.14159265358979323846F;

[[nodiscard]] constexpr float radians(float degrees) noexcept {
    return degrees * (PI_F / 180.0F);
}

struct Vec3 {
    float x{0.0F};
    float y{0.0F};
    float z{0.0F};

    constexpr Vec3() = default;
    constexpr Vec3(float x, float y, float z) noexcept : x(x), y(y), z(z) {}
};

[[nodiscard]] constexpr Vec3 operator+(Vec3 lhs, Vec3 rhs) noexcept {
    return {lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z};
}
[[nodiscard]] constexpr Vec3 operator-(Vec3 lhs, Vec3 rhs) noexcept {
    return {lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z};
}
[[nodiscard]] constexpr Vec3 operator-(Vec3 value) noexcept {
    return {-value.x, -value.y, -value.z};
}
[[nodiscard]] constexpr Vec3 operator*(Vec3 value, float scalar) noexcept {
    return {value.x * scalar, value.y * scalar, value.z * scalar};
}
[[nodiscard]] constexpr Vec3 operator*(float scalar, Vec3 value) noexcept {
    return value * scalar;
}
[[nodiscard]] constexpr Vec3 hadamard(Vec3 lhs, Vec3 rhs) noexcept {
    return {lhs.x * rhs.x, lhs.y * rhs.y, lhs.z * rhs.z};
}

[[nodiscard]] constexpr float dot(Vec3 lhs, Vec3 rhs) noexcept {
    return (lhs.x * rhs.x) + (lhs.y * rhs.y) + (lhs.z * rhs.z);
}

[[nodiscard]] constexpr Vec3 cross(Vec3 lhs, Vec3 rhs) noexcept {
    return {(lhs.y * rhs.z) - (lhs.z * rhs.y),
            (lhs.z * rhs.x) - (lhs.x * rhs.z),
            (lhs.x * rhs.y) - (lhs.y * rhs.x)};
}

[[nodiscard]] inline float length(Vec3 value) noexcept {
    return std::sqrt(dot(value, value));
}

[[nodiscard]] inline Vec3 normalize(Vec3 value) noexcept {
    const float len = length(value);
    if (len <= 0.0F) {
        return {0.0F, 0.0F, 0.0F};
    }
    return value * (1.0F / len);
}

struct Vec4 {
    float x{0.0F};
    float y{0.0F};
    float z{0.0F};
    float w{0.0F};

    constexpr Vec4() = default;
    constexpr Vec4(float x, float y, float z, float w_value) noexcept
        : x(x), y(y), z(z), w(w_value) {}
};

// Column-major 4x4 matrix: element (row, col) lives at m[(col * 4) + row].
struct Mat4 {
    std::array<float, 16> data{};

    [[nodiscard]] static constexpr Mat4 identity() noexcept {
        Mat4 out{};
        out.data.at(0) = 1.0F;
        out.data.at(5) = 1.0F;
        out.data.at(10) = 1.0F;
        out.data.at(15) = 1.0F;
        return out;
    }

    [[nodiscard]] constexpr float at(std::size_t row, std::size_t col) const noexcept {
        return data.at((col * 4) + row);
    }
    constexpr void set(std::size_t row, std::size_t col, float value) noexcept {
        data.at((col * 4) + row) = value;
    }
};

[[nodiscard]] constexpr Mat4 operator*(const Mat4& lhs, const Mat4& rhs) noexcept {
    Mat4 out{};
    for (std::size_t col = 0; col < 4; ++col) {
        for (std::size_t row = 0; row < 4; ++row) {
            float sum = 0.0F;
            for (std::size_t k = 0; k < 4; ++k) {
                sum += lhs.at(row, k) * rhs.at(k, col);
            }
            out.set(row, col, sum);
        }
    }
    return out;
}

// Transform a homogeneous point (w assumed 1) by a matrix.
[[nodiscard]] constexpr Vec4 transform(const Mat4& mat, Vec3 point) noexcept {
    const std::array<float, 4> vec{point.x, point.y, point.z, 1.0F};
    std::array<float, 4> out{};
    for (std::size_t row = 0; row < 4; ++row) {
        float sum = 0.0F;
        for (std::size_t col = 0; col < 4; ++col) {
            sum += mat.at(row, col) * vec.at(col);
        }
        out.at(row) = sum;
    }
    return {out.at(0), out.at(1), out.at(2), out.at(3)};
}

[[nodiscard]] constexpr Mat4 translation(Vec3 offset) noexcept {
    Mat4 out = Mat4::identity();
    out.set(0, 3, offset.x);
    out.set(1, 3, offset.y);
    out.set(2, 3, offset.z);
    return out;
}

[[nodiscard]] constexpr Mat4 scaling(Vec3 factor) noexcept {
    Mat4 out{};
    out.set(0, 0, factor.x);
    out.set(1, 1, factor.y);
    out.set(2, 2, factor.z);
    out.set(3, 3, 1.0F);
    return out;
}

// Right-handed look-at view matrix (camera looks down -Z, OpenGL style).
[[nodiscard]] inline Mat4 look_at(Vec3 eye, Vec3 center, Vec3 up_dir) noexcept {
    const Vec3 forward = normalize(center - eye);
    const Vec3 right = normalize(cross(forward, up_dir));
    const Vec3 true_up = cross(right, forward);

    Mat4 out = Mat4::identity();
    out.set(0, 0, right.x);
    out.set(0, 1, right.y);
    out.set(0, 2, right.z);
    out.set(1, 0, true_up.x);
    out.set(1, 1, true_up.y);
    out.set(1, 2, true_up.z);
    out.set(2, 0, -forward.x);
    out.set(2, 1, -forward.y);
    out.set(2, 2, -forward.z);
    out.set(0, 3, -dot(right, eye));
    out.set(1, 3, -dot(true_up, eye));
    out.set(2, 3, dot(forward, eye));
    return out;
}

// Right-handed perspective projection mapping depth to [-1, 1].
[[nodiscard]] inline Mat4 perspective(float fov_y_radians, float aspect, float z_near,
                                      float z_far) noexcept {
    const float tan_half = std::tan(fov_y_radians * 0.5F);
    Mat4 out{};
    out.set(0, 0, 1.0F / (aspect * tan_half));
    out.set(1, 1, 1.0F / tan_half);
    out.set(2, 2, -(z_far + z_near) / (z_far - z_near));
    out.set(2, 3, -(2.0F * z_far * z_near) / (z_far - z_near));
    out.set(3, 2, -1.0F);
    return out;
}

// Inverse-transpose of the upper-left 3x3, used to transform normals under
// non-uniform scaling. Returns the identity-ish direction unchanged if the
// matrix is singular.
[[nodiscard]] Vec3 transform_normal(const Mat4& model, Vec3 normal) noexcept;

}  // namespace pacd::render
