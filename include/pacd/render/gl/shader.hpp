#pragma once

#include <expected>
#include <string>
#include <string_view>

#include "pacd/render/math.hpp"

namespace pacd::render {

// RAII wrapper around a linked GLSL program. OpenGL types are intentionally
// kept out of this header (the program handle is a plain unsigned int) so that
// translation units which only orchestrate rendering need not include glad.
class ShaderProgram {
   public:
    ShaderProgram() = default;
    ~ShaderProgram();
    ShaderProgram(const ShaderProgram&) = delete;
    ShaderProgram& operator=(const ShaderProgram&) = delete;
    ShaderProgram(ShaderProgram&& other) noexcept;
    ShaderProgram& operator=(ShaderProgram&& other) noexcept;

    // Compile and link the given sources. Returns an error string on failure.
    [[nodiscard]] std::expected<void, std::string> build(std::string_view vertex_source,
                                                         std::string_view fragment_source);

    void use() const noexcept;
    void set_mat4(std::string_view name, const Mat4& value) const;
    void set_vec3(std::string_view name, Vec3 value) const;
    void set_float(std::string_view name, float value) const;
    void set_int(std::string_view name, int value) const;

    [[nodiscard]] unsigned int id() const noexcept { return m_program; }

   private:
    unsigned int m_program{0};
};

}  // namespace pacd::render
