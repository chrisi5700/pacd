#pragma once

#include <string>
#include <string_view>
#include <vector>

#include "pacd/render/camera.hpp"
#include "pacd/render/gl/gpu_mesh.hpp"
#include "pacd/render/gl/shader.hpp"
#include "pacd/render/scene.hpp"

namespace pacd::render {

struct ViewerOptions {
    int width{1280};
    int height{800};
    std::string title{"pacd viewer"};
    bool visible{true};  // false => offscreen window (screenshots / CI)
    int msaa_samples{4};
};

// Interactive OpenGL viewer: owns a GLFW window and GL context, renders a Scene
// with Blinn-Phong shading, and drives an OrbitCamera from the mouse
// (left-drag orbit, middle/right-drag pan, scroll to zoom). Keyboard: F frame,
// W wireframe, S screenshot, Esc quit.
//
// The window handle is type-erased to keep GLFW/OpenGL out of this header.
class Viewer {
   public:
    explicit Viewer(ViewerOptions options = {});
    ~Viewer();
    Viewer(const Viewer&) = delete;
    Viewer& operator=(const Viewer&) = delete;
    Viewer(Viewer&&) = delete;
    Viewer& operator=(Viewer&&) = delete;

    [[nodiscard]] bool valid() const noexcept;
    [[nodiscard]] const std::string& error() const noexcept { return m_error; }

    void set_scene(Scene scene);
    [[nodiscard]] Scene& scene() noexcept { return m_scene; }
    [[nodiscard]] OrbitCamera& camera() noexcept { return m_camera; }
    void frame_scene() noexcept;
    void set_wireframe(bool enabled) noexcept { m_wireframe = enabled; }

    void run();                 // blocking interactive loop until the window closes
    void run_frames(int count);  // render a fixed number of frames (headless smoke test)
    [[nodiscard]] bool save_screenshot(std::string_view path) const;

    // Invoked by the GLFW C callbacks; not intended for direct use.
    void handle_cursor(double xpos, double ypos);
    void handle_button(int button, int action, double xpos, double ypos);
    void handle_scroll(double y_offset);
    void handle_key(int key, int action);
    void handle_resize(int width, int height);

   private:
    void render_frame();
    void apply_light_uniforms();
    void draw_objects();

    void* m_window{nullptr};
    bool m_glfw_ready{false};
    std::string m_error;
    ViewerOptions m_options;

    ShaderProgram m_shader;
    std::vector<GpuMesh> m_gpu;
    Scene m_scene;
    OrbitCamera m_camera;

    double m_last_x{0.0};
    double m_last_y{0.0};
    bool m_orbit_active{false};
    bool m_pan_active{false};
    int m_fb_width{0};
    int m_fb_height{0};
    bool m_wireframe{false};
    int m_screenshot_index{0};
};

}  // namespace pacd::render
