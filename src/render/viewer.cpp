#include "pacd/render/viewer.hpp"

#include <glad/glad.h>
// GLFW must follow glad.
#include <GLFW/glfw3.h>

// Dear ImGui + its GLFW/OpenGL3 backends (GL stack must already be included).
#include <algorithm>
#include <cstddef>
#include <filesystem>
#include <functional>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#include "pacd/render/camera.hpp"
#include "pacd/render/gl/gpu_mesh.hpp"
#include "pacd/render/gl/shader.hpp"
#include "pacd/render/image.hpp"
#include "pacd/render/math.hpp"
#include "pacd/render/scene.hpp"

namespace pacd::render
{

namespace
{

constexpr float		  ORBIT_SPEED = 0.008F;
constexpr std::size_t MAX_LIGHTS  = 4;

// Interactive screenshots (the S key) collect here, next to the repo's showcase.
constexpr std::string_view SCREENSHOT_DIR = "img";

constexpr std::string_view VERTEX_SRC = R"(#version 330 core
layout(location = 0) in vec3 a_position;
layout(location = 1) in vec3 a_normal;
uniform mat4 u_model;
uniform mat4 u_view;
uniform mat4 u_proj;
out vec3 v_world_pos;
out vec3 v_normal;
void main() {
    vec4 world = u_model * vec4(a_position, 1.0);
    v_world_pos = world.xyz;
    v_normal = mat3(transpose(inverse(u_model))) * a_normal;
    gl_Position = u_proj * u_view * world;
}
)";

constexpr std::string_view FRAGMENT_SRC = R"(#version 330 core
in vec3 v_world_pos;
in vec3 v_normal;
out vec4 frag_color;
uniform vec3 u_view_pos;
uniform vec3 u_albedo;
uniform vec3 u_ambient;
uniform float u_specular_strength;
uniform float u_shininess;
uniform int u_light_count;
uniform vec3 u_light_dir[4];
uniform vec3 u_light_color[4];
uniform int u_wireframe;
void main() {
    if (u_wireframe == 1) {
        frag_color = vec4(u_albedo, 1.0);
        return;
    }
    vec3 normal = normalize(v_normal);
    vec3 view_dir = normalize(u_view_pos - v_world_pos);
    if (dot(normal, view_dir) < 0.0) {
        normal = -normal;  // two-sided shading for open/CAD meshes
    }
    vec3 color = u_ambient * u_albedo;
    for (int i = 0; i < u_light_count; ++i) {
        vec3 light_dir = normalize(-u_light_dir[i]);
        float diffuse = max(dot(normal, light_dir), 0.0);
        vec3 half_vec = normalize(light_dir + view_dir);
        float specular = pow(max(dot(normal, half_vec), 0.0), u_shininess) * u_specular_strength;
        color += u_light_color[i] * (u_albedo * diffuse + vec3(specular));
    }
    frag_color = vec4(color, 1.0);
}
)";

[[nodiscard]] Viewer* viewer_of(GLFWwindow* window)
{
	return static_cast<Viewer*>(glfwGetWindowUserPointer(window));
}

void cursor_callback(GLFWwindow* window, double xpos, double ypos)
{
	if (Viewer* viewer = viewer_of(window))
	{
		viewer->handle_cursor(xpos, ypos);
	}
}

void button_callback(GLFWwindow* window, int button, int action, int /*mods*/)
{
	double xpos = 0.0;
	double ypos = 0.0;
	glfwGetCursorPos(window, &xpos, &ypos);
	if (Viewer* viewer = viewer_of(window))
	{
		viewer->handle_button(button, action, xpos, ypos);
	}
}

void scroll_callback(GLFWwindow* window, double /*xoffset*/, double yoffset)
{
	if (Viewer* viewer = viewer_of(window))
	{
		viewer->handle_scroll(yoffset);
	}
}

void key_callback(GLFWwindow* window, int key, int /*scancode*/, int action, int /*mods*/)
{
	if (Viewer* viewer = viewer_of(window))
	{
		viewer->handle_key(key, action);
	}
}

void resize_callback(GLFWwindow* window, int width, int height)
{
	if (Viewer* viewer = viewer_of(window))
	{
		viewer->handle_resize(width, height);
	}
}

} // namespace

Viewer::Viewer(ViewerOptions options)
	: m_options(std::move(options))
{
	if (glfwInit() == GLFW_FALSE)
	{
		m_error = "glfwInit failed (no display?)";
		return;
	}
	m_glfw_ready = true;
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	glfwWindowHint(GLFW_SAMPLES, m_options.msaa_samples);
	glfwWindowHint(GLFW_VISIBLE, m_options.visible ? GLFW_TRUE : GLFW_FALSE);

	GLFWwindow* window = glfwCreateWindow(m_options.width, m_options.height, m_options.title.c_str(), nullptr, nullptr);
	if (window == nullptr)
	{
		m_error = "failed to create GLFW window / GL 3.3 context";
		return;
	}
	m_window = window;
	glfwMakeContextCurrent(window);
	glfwSwapInterval(1);

	if (gladLoadGLLoader(reinterpret_cast<GLADloadproc>(glfwGetProcAddress)) == 0)
	{
		m_error = "failed to load OpenGL functions";
		return;
	}

	glfwSetWindowUserPointer(window, this);
	glfwSetCursorPosCallback(window, cursor_callback);
	glfwSetMouseButtonCallback(window, button_callback);
	glfwSetScrollCallback(window, scroll_callback);
	glfwSetKeyCallback(window, key_callback);
	glfwSetFramebufferSizeCallback(window, resize_callback);
	glfwGetFramebufferSize(window, &m_fb_width, &m_fb_height);

	glEnable(GL_DEPTH_TEST);
	glEnable(GL_MULTISAMPLE);

	const std::expected<void, std::string> built = m_shader.build(VERTEX_SRC, FRAGMENT_SRC);
	if (!built)
	{
		m_error = built.error();
		return;
	}

	// Dear ImGui for the interactive config panel. The panel is only *drawn* when
	// the app installs an on_gui callback, so headless/overlay screenshots (which
	// don't) stay panel-free -- but the context is created unconditionally so an
	// offscreen framebuffer can still snapshot a panel when asked. The GLFW
	// backend is installed with chaining (true), so it forwards events to the
	// input callbacks registered above after updating its own IO.
	IMGUI_CHECKVERSION();
	ImGui::CreateContext();
	ImGui::GetIO().IniFilename = nullptr; // don't litter an imgui.ini
	ImGui::StyleColorsDark();
	ImGui_ImplGlfw_InitForOpenGL(window, true);
	ImGui_ImplOpenGL3_Init("#version 330 core");
	m_imgui_ready = true;
}

Viewer::~Viewer()
{
	if (m_window != nullptr)
	{
		// Tear down ImGui while the GL context is still current.
		if (m_imgui_ready)
		{
			ImGui_ImplOpenGL3_Shutdown();
			ImGui_ImplGlfw_Shutdown();
			ImGui::DestroyContext();
		}
		// Release GL objects while the context is still current.
		m_gpu.clear();
		m_shader = ShaderProgram{};
		glfwDestroyWindow(static_cast<GLFWwindow*>(m_window));
		m_window = nullptr;
	}
	if (m_glfw_ready)
	{
		glfwTerminate();
	}
}

bool Viewer::valid() const noexcept
{
	return m_window != nullptr && m_error.empty() && m_shader.id() != 0;
}

void Viewer::set_scene(Scene new_scene)
{
	m_scene = std::move(new_scene);
	m_gpu.clear();
	m_gpu.reserve(m_scene.objects.size());
	for (const Object& object : m_scene.objects)
	{
		GpuMesh gpu;
		gpu.upload(object.mesh);
		m_gpu.push_back(std::move(gpu));
	}
	frame_scene();
}

void Viewer::frame_scene() noexcept
{
	m_camera.frame(m_scene.bounds());
}

void Viewer::set_on_cycle(std::function<void(int)> callback)
{
	m_on_cycle = std::move(callback);
}

void Viewer::set_on_toggle_original(std::function<void()> callback)
{
	m_on_toggle = std::move(callback);
}

void Viewer::set_on_redecompose(std::function<void()> callback)
{
	m_on_redecompose = std::move(callback);
}

void Viewer::set_on_gui(std::function<void()> callback)
{
	m_on_gui = std::move(callback);
}

bool Viewer::should_close() const noexcept
{
	return m_window == nullptr || glfwWindowShouldClose(static_cast<GLFWwindow*>(m_window)) != 0;
}

bool Viewer::pump()
{
	if (should_close())
	{
		return false;
	}
	glfwPollEvents(); // may fire input callbacks (cycle / toggle / re-decompose)
	draw_once();	  // render + present one frame (incl. the ImGui panel)
	return true;
}

void Viewer::draw_once()
{
	if (!valid())
	{
		return;
	}
	render_frame();
	glfwSwapBuffers(static_cast<GLFWwindow*>(m_window));
}

void Viewer::apply_light_uniforms()
{
	const std::size_t count = std::min(m_scene.lights.size(), MAX_LIGHTS);
	m_shader.set_int("u_light_count", static_cast<int>(count));
	for (std::size_t light = 0; light < count; ++light)
	{
		const std::string		suffix = "[" + std::to_string(light) + "]";
		const DirectionalLight& source = m_scene.lights.at(light);
		m_shader.set_vec3("u_light_dir" + suffix, normalize(source.direction));
		m_shader.set_vec3("u_light_color" + suffix, source.color * source.intensity);
	}
}

void Viewer::draw_objects()
{
	for (std::size_t index = 0; index < m_scene.objects.size() && index < m_gpu.size(); ++index)
	{
		const Object& object = m_scene.objects.at(index);
		if (!object.visible)
		{
			continue;
		}
		const bool wire = m_wireframe || object.wireframe;
		glPolygonMode(GL_FRONT_AND_BACK, wire ? GL_LINE : GL_FILL);
		m_shader.set_int("u_wireframe", wire ? 1 : 0);
		m_shader.set_mat4("u_model", object.model);
		m_shader.set_vec3("u_albedo", object.material.albedo);
		m_shader.set_float("u_specular_strength", object.material.specular_strength);
		m_shader.set_float("u_shininess", object.material.shininess);
		m_gpu.at(index).draw();
	}
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
}

void Viewer::render_frame()
{
	auto* window = static_cast<GLFWwindow*>(m_window);
	glfwGetFramebufferSize(window, &m_fb_width, &m_fb_height);
	glViewport(0, 0, m_fb_width, m_fb_height);

	const Vec3 background = m_scene.background;
	glClearColor(background.x, background.y, background.z, 1.0F);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	const float aspect = static_cast<float>(std::max(m_fb_width, 1)) / static_cast<float>(std::max(m_fb_height, 1));
	m_shader.use();
	m_shader.set_mat4("u_view", m_camera.view());
	m_shader.set_mat4("u_proj", m_camera.projection(aspect));
	m_shader.set_vec3("u_view_pos", m_camera.eye());
	m_shader.set_vec3("u_ambient", m_scene.ambient);
	apply_light_uniforms();
	draw_objects();

	// Overlay the app's ImGui panel (if any) on top of the scene.
	if (m_imgui_ready && m_on_gui)
	{
		ImGui_ImplOpenGL3_NewFrame();
		ImGui_ImplGlfw_NewFrame();
		ImGui::NewFrame();
		m_on_gui();
		ImGui::Render();
		ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
	}
}

void Viewer::run()
{
	if (!valid())
	{
		return;
	}
	auto* window = static_cast<GLFWwindow*>(m_window);
	while (glfwWindowShouldClose(window) == 0)
	{
		render_frame();
		glfwSwapBuffers(window);
		glfwPollEvents();
	}
}

void Viewer::run_frames(int count)
{
	if (!valid())
	{
		return;
	}
	auto* window = static_cast<GLFWwindow*>(m_window);
	for (int frame = 0; frame < count && glfwWindowShouldClose(window) == 0; ++frame)
	{
		render_frame();
		glfwSwapBuffers(window);
		glfwPollEvents();
	}
}

bool Viewer::save_screenshot(std::string_view path) const
{
	if (m_window == nullptr)
	{
		return false;
	}
	int width  = 0;
	int height = 0;
	glfwGetFramebufferSize(static_cast<GLFWwindow*>(m_window), &width, &height);
	if (width <= 0 || height <= 0)
	{
		return false;
	}
	const auto				  pixel_count = static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
	std::vector<std::uint8_t> buffer(pixel_count * 3);
	glPixelStorei(GL_PACK_ALIGNMENT, 1);
	glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, buffer.data());

	Image			image(width, height);
	constexpr float INV = 1.0F / 255.0F;
	for (int row = 0; row < height; ++row)
	{
		const int flipped = height - 1 - row; // OpenGL origin is bottom-left
		for (int col = 0; col < width; ++col)
		{
			const std::size_t base =
				((static_cast<std::size_t>(row) * static_cast<std::size_t>(width)) + static_cast<std::size_t>(col)) * 3;
			image.set_pixel(col, flipped,
							{static_cast<float>(buffer.at(base + 0)) * INV,
							 static_cast<float>(buffer.at(base + 1)) * INV,
							 static_cast<float>(buffer.at(base + 2)) * INV});
		}
	}
	return image.save_png(path);
}

void Viewer::handle_cursor(double xpos, double ypos)
{
	const double delta_x = xpos - m_last_x;
	const double delta_y = ypos - m_last_y;
	m_last_x			 = xpos;
	m_last_y			 = ypos;
	if (m_orbit_active)
	{
		m_camera.orbit(static_cast<float>(delta_x) * ORBIT_SPEED, static_cast<float>(-delta_y) * ORBIT_SPEED);
	}
	else if (m_pan_active)
	{
		m_camera.pan(static_cast<float>(delta_x), static_cast<float>(delta_y),
					 static_cast<float>(std::max(m_fb_height, 1)));
	}
}

void Viewer::handle_button(int button, int action, double xpos, double ypos)
{
	// Let the panel consume clicks landing on it (don't start an orbit/pan).
	if (m_imgui_ready && ImGui::GetIO().WantCaptureMouse)
	{
		return;
	}
	const bool pressed = (action == GLFW_PRESS);
	if (button == GLFW_MOUSE_BUTTON_LEFT)
	{
		m_orbit_active = pressed;
	}
	else if (button == GLFW_MOUSE_BUTTON_MIDDLE || button == GLFW_MOUSE_BUTTON_RIGHT)
	{
		m_pan_active = pressed;
	}
	if (pressed)
	{
		m_last_x = xpos;
		m_last_y = ypos;
	}
}

void Viewer::handle_scroll(double y_offset)
{
	if (m_imgui_ready && ImGui::GetIO().WantCaptureMouse)
	{
		return; // scrolling over the panel adjusts widgets, not the camera
	}
	m_camera.dolly(static_cast<float>(y_offset));
}

void Viewer::handle_key(int key, int action)
{
	if (action != GLFW_PRESS)
	{
		return;
	}
	if (m_imgui_ready && ImGui::GetIO().WantCaptureKeyboard)
	{
		return; // a focused ImGui field owns the keyboard
	}
	auto* window = static_cast<GLFWwindow*>(m_window);
	if (key == GLFW_KEY_ESCAPE)
	{
		glfwSetWindowShouldClose(window, GLFW_TRUE);
	}
	else if (key == GLFW_KEY_F)
	{
		frame_scene();
	}
	else if (key == GLFW_KEY_W)
	{
		m_wireframe = !m_wireframe;
	}
	else if (key == GLFW_KEY_S)
	{
		// Collect shots in the showcase dir, and skip past any left by earlier
		// sessions so restarting appends instead of overwriting shot 0 onward.
		const std::filesystem::path dir{SCREENSHOT_DIR};
		std::error_code				mkdir_ec;
		std::filesystem::create_directories(dir, mkdir_ec);
		auto shot_path = [&dir](int index) { return dir / ("pacd-shot-" + std::to_string(index) + ".png"); };
		std::filesystem::path path = shot_path(m_screenshot_index);
		while (std::filesystem::exists(path))
		{
			++m_screenshot_index;
			path = shot_path(m_screenshot_index);
		}
		++m_screenshot_index;
		std::ignore = save_screenshot(path.string());
	}
	else if (key == GLFW_KEY_LEFT)
	{
		if (m_on_cycle)
		{
			m_on_cycle(-1);
		}
	}
	else if (key == GLFW_KEY_RIGHT)
	{
		if (m_on_cycle)
		{
			m_on_cycle(1);
		}
	}
	else if (key == GLFW_KEY_T)
	{
		if (m_on_toggle)
		{
			m_on_toggle();
		}
	}
	else if (key == GLFW_KEY_R)
	{
		if (m_on_redecompose)
		{
			m_on_redecompose();
		}
	}
}

void Viewer::handle_resize(int width, int height)
{
	m_fb_width	= width;
	m_fb_height = height;
}

} // namespace pacd::render
