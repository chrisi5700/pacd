#pragma once

#include <cstddef>

namespace pacd::render {

struct Mesh;

// RAII owner of the OpenGL buffers for one indexed mesh. Positions and normals
// live in separate VBOs so each vertex attribute starts at offset zero, which
// keeps the upload free of pointer-offset casts.
class GpuMesh {
   public:
    GpuMesh() = default;
    ~GpuMesh();
    GpuMesh(const GpuMesh&) = delete;
    GpuMesh& operator=(const GpuMesh&) = delete;
    GpuMesh(GpuMesh&& other) noexcept;
    GpuMesh& operator=(GpuMesh&& other) noexcept;

    void upload(const Mesh& mesh);
    void draw() const noexcept;

    [[nodiscard]] std::size_t index_count() const noexcept { return m_index_count; }

   private:
    void release() noexcept;

    unsigned int m_vao{0};
    unsigned int m_position_vbo{0};
    unsigned int m_normal_vbo{0};
    unsigned int m_ebo{0};
    std::size_t m_index_count{0};
};

}  // namespace pacd::render
