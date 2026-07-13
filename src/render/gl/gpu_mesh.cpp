#include "pacd/render/gl/gpu_mesh.hpp"

#include <cstddef>
#include <glad/glad.h>
#include <utility>
#include <vector>

#include "pacd/render/math.hpp"
#include "pacd/render/mesh.hpp"

namespace pacd::render
{

namespace
{
constexpr GLuint POSITION_LOCATION = 0;
constexpr GLuint NORMAL_LOCATION   = 1;

[[nodiscard]] GLsizeiptr byte_size(std::size_t count, std::size_t element) noexcept
{
	return static_cast<GLsizeiptr>(count * element);
}
} // namespace

GpuMesh::~GpuMesh()
{
	release();
}

GpuMesh::GpuMesh(GpuMesh&& other) noexcept
	: m_vao(std::exchange(other.m_vao, 0U))
	, m_position_vbo(std::exchange(other.m_position_vbo, 0U))
	, m_normal_vbo(std::exchange(other.m_normal_vbo, 0U))
	, m_ebo(std::exchange(other.m_ebo, 0U))
	, m_index_count(std::exchange(other.m_index_count, 0U))
{
}

GpuMesh& GpuMesh::operator=(GpuMesh&& other) noexcept
{
	if (this != &other)
	{
		release();
		m_vao		   = std::exchange(other.m_vao, 0U);
		m_position_vbo = std::exchange(other.m_position_vbo, 0U);
		m_normal_vbo   = std::exchange(other.m_normal_vbo, 0U);
		m_ebo		   = std::exchange(other.m_ebo, 0U);
		m_index_count  = std::exchange(other.m_index_count, 0U);
	}
	return *this;
}

void GpuMesh::release() noexcept
{
	if (m_ebo != 0)
	{
		glDeleteBuffers(1, &m_ebo);
	}
	if (m_normal_vbo != 0)
	{
		glDeleteBuffers(1, &m_normal_vbo);
	}
	if (m_position_vbo != 0)
	{
		glDeleteBuffers(1, &m_position_vbo);
	}
	if (m_vao != 0)
	{
		glDeleteVertexArrays(1, &m_vao);
	}
	m_vao		   = 0;
	m_position_vbo = 0;
	m_normal_vbo   = 0;
	m_ebo		   = 0;
	m_index_count  = 0;
}

void GpuMesh::upload(const Mesh& mesh)
{
	release();

	// Guarantee a normal per position so attribute 1 is always well-defined.
	std::vector<Vec3> normals = mesh.normals;
	normals.resize(mesh.positions.size(), Vec3{0.0F, 1.0F, 0.0F});

	glGenVertexArrays(1, &m_vao);
	glBindVertexArray(m_vao);

	glGenBuffers(1, &m_position_vbo);
	glBindBuffer(GL_ARRAY_BUFFER, m_position_vbo);
	glBufferData(GL_ARRAY_BUFFER, byte_size(mesh.positions.size(), sizeof(Vec3)), mesh.positions.data(),
				 GL_STATIC_DRAW);
	glEnableVertexAttribArray(POSITION_LOCATION);
	glVertexAttribPointer(POSITION_LOCATION, 3, GL_FLOAT, GL_FALSE, sizeof(Vec3), nullptr);

	glGenBuffers(1, &m_normal_vbo);
	glBindBuffer(GL_ARRAY_BUFFER, m_normal_vbo);
	glBufferData(GL_ARRAY_BUFFER, byte_size(normals.size(), sizeof(Vec3)), normals.data(), GL_STATIC_DRAW);
	glEnableVertexAttribArray(NORMAL_LOCATION);
	glVertexAttribPointer(NORMAL_LOCATION, 3, GL_FLOAT, GL_FALSE, sizeof(Vec3), nullptr);

	glGenBuffers(1, &m_ebo);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_ebo);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, byte_size(mesh.triangles.size(), sizeof(IndexTriangle)),
				 mesh.triangles.data(), GL_STATIC_DRAW);

	glBindVertexArray(0);
	m_index_count = mesh.triangles.size() * 3;
}

void GpuMesh::draw() const noexcept
{
	if (m_vao == 0 || m_index_count == 0)
	{
		return;
	}
	glBindVertexArray(m_vao);
	glDrawElements(GL_TRIANGLES, static_cast<GLsizei>(m_index_count), GL_UNSIGNED_INT, nullptr);
	glBindVertexArray(0);
}

} // namespace pacd::render
