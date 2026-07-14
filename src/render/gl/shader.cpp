#include "pacd/render/gl/shader.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <expected>
#include <glad/glad.h>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "pacd/render/math.hpp"

namespace pacd::render
{

namespace
{

[[nodiscard]] std::string shader_log(GLuint shader)
{
	GLint log_length = 0;
	glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &log_length);
	std::vector<char> log(static_cast<std::size_t>(std::max(log_length, 1)));
	glGetShaderInfoLog(shader, static_cast<GLsizei>(log.size()), nullptr, log.data());
	return std::string{log.data()};
}

[[nodiscard]] std::string program_log(GLuint program)
{
	GLint log_length = 0;
	glGetProgramiv(program, GL_INFO_LOG_LENGTH, &log_length);
	std::vector<char> log(static_cast<std::size_t>(std::max(log_length, 1)));
	glGetProgramInfoLog(program, static_cast<GLsizei>(log.size()), nullptr, log.data());
	return std::string{log.data()};
}

[[nodiscard]] std::expected<GLuint, std::string> compile(GLenum stage, std::string_view source)
{
	const GLuint					 shader = glCreateShader(stage);
	const std::string				 text{source};
	const std::array<const char*, 1> sources{text.c_str()};
	const std::array<GLint, 1>		 lengths{static_cast<GLint>(text.size())};
	glShaderSource(shader, 1, sources.data(), lengths.data());
	glCompileShader(shader);
	GLint ok = GL_FALSE;
	glGetShaderiv(shader, GL_COMPILE_STATUS, &ok);
	if (ok == GL_FALSE)
	{
		const std::string message = shader_log(shader);
		glDeleteShader(shader);
		return std::unexpected(std::string{"shader compile failed: "} + message);
	}
	return shader;
}

} // namespace

ShaderProgram::~ShaderProgram()
{
	if (m_program != 0)
	{
		glDeleteProgram(m_program);
	}
}

ShaderProgram::ShaderProgram(ShaderProgram&& other) noexcept
	: m_program(std::exchange(other.m_program, 0U))
{
}

ShaderProgram& ShaderProgram::operator=(ShaderProgram&& other) noexcept
{
	if (this != &other)
	{
		if (m_program != 0)
		{
			glDeleteProgram(m_program);
		}
		m_program = std::exchange(other.m_program, 0U);
	}
	return *this;
}

std::expected<void, std::string> ShaderProgram::build(std::string_view vertex_source, std::string_view fragment_source)
{
	const std::expected<GLuint, std::string> vertex = compile(GL_VERTEX_SHADER, vertex_source);
	if (!vertex)
	{
		return std::unexpected(vertex.error());
	}
	const std::expected<GLuint, std::string> fragment = compile(GL_FRAGMENT_SHADER, fragment_source);
	if (!fragment)
	{
		glDeleteShader(*vertex);
		return std::unexpected(fragment.error());
	}

	const GLuint program = glCreateProgram();
	glAttachShader(program, *vertex);
	glAttachShader(program, *fragment);
	glLinkProgram(program);
	glDeleteShader(*vertex);
	glDeleteShader(*fragment);

	GLint ok = GL_FALSE;
	glGetProgramiv(program, GL_LINK_STATUS, &ok);
	if (ok == GL_FALSE)
	{
		const std::string message = program_log(program);
		glDeleteProgram(program);
		return std::unexpected(std::string{"program link failed: "} + message);
	}

	if (m_program != 0)
	{
		glDeleteProgram(m_program);
	}
	m_program = program;
	return {};
}

void ShaderProgram::use() const noexcept
{
	glUseProgram(m_program);
}

void ShaderProgram::set_mat4(std::string_view name, const Mat4& value) const
{
	const GLint location = glGetUniformLocation(m_program, std::string{name}.c_str());
	glUniformMatrix4fv(location, 1, GL_FALSE, value.data.data());
}

void ShaderProgram::set_vec3(std::string_view name, Vec3 value) const
{
	const GLint location = glGetUniformLocation(m_program, std::string{name}.c_str());
	glUniform3f(location, value.x, value.y, value.z);
}

void ShaderProgram::set_float(std::string_view name, float value) const
{
	const GLint location = glGetUniformLocation(m_program, std::string{name}.c_str());
	glUniform1f(location, value);
}

void ShaderProgram::set_int(std::string_view name, int value) const
{
	const GLint location = glGetUniformLocation(m_program, std::string{name}.c_str());
	glUniform1i(location, value);
}

} // namespace pacd::render
