#include "pacd/render/scenes.hpp"

#include <expected>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "pacd/render/math.hpp"
#include "pacd/render/mesh.hpp"
#include "pacd/render/primitives.hpp"
#include "pacd/render/scene.hpp"
#include "pacd/render/stl.hpp"

namespace pacd::render
{

const std::vector<Vec3>& palette()
{
	static const std::vector<Vec3> COLORS{
		{0.86F, 0.36F, 0.31F}, {0.34F, 0.64F, 0.86F}, {0.46F, 0.80F, 0.46F}, {0.91F, 0.76F, 0.35F},
		{0.70F, 0.51F, 0.86F}, {0.86F, 0.56F, 0.35F}, {0.42F, 0.80F, 0.80F}, {0.87F, 0.46F, 0.71F},
	};
	return COLORS;
}

namespace
{

Object make_object(Mesh mesh, Vec3 position, Vec3 color)
{
	Object object;
	object.mesh			   = std::move(mesh);
	object.model		   = translation(position);
	object.material.albedo = color;
	return object;
}

} // namespace

Scene make_primitive_gallery()
{
	Scene scene;
	scene.lights					= default_lights();
	const std::vector<Vec3>& colors = palette();
	scene.objects.push_back(make_object(make_box({1.0F, 1.0F, 1.0F}), {-3.0F, 0.5F, 0.0F}, colors.at(0)));
	scene.objects.push_back(make_object(make_uv_sphere(0.5F), {-1.0F, 0.5F, 0.0F}, colors.at(1)));
	scene.objects.push_back(make_object(make_cylinder(0.5F, 1.0F), {1.0F, 0.5F, 0.0F}, colors.at(2)));
	scene.objects.push_back(make_object(make_cone(0.5F, 1.0F), {3.0F, 0.5F, 0.0F}, colors.at(3)));

	Object ground					  = make_object(make_plane(12.0F, 6.0F), {0.0F, 0.0F, 0.0F}, {0.55F, 0.55F, 0.58F});
	ground.material.specular_strength = 0.05F;
	scene.objects.push_back(std::move(ground));
	return scene;
}

std::expected<Scene, std::string> load_stl_scene(std::span<const std::string> paths, bool weld)
{
	Scene scene;
	scene.lights					= default_lights();
	const std::vector<Vec3>& colors = palette();
	std::size_t				 index	= 0;
	for (const std::string& path : paths)
	{
		std::expected<Mesh, std::string> mesh = load_stl(path);
		if (!mesh)
		{
			return std::unexpected(mesh.error());
		}
		Mesh	   geometry = weld ? with_crease_normals(*mesh) : std::move(*mesh);
		const Vec3 color	= colors.at(index % colors.size());
		Object	   object;
		object.mesh			   = std::move(geometry);
		object.material.albedo = color;
		scene.objects.push_back(std::move(object));
		++index;
	}
	if (scene.objects.empty())
	{
		return std::unexpected(std::string{"no meshes loaded"});
	}
	return scene;
}

} // namespace pacd::render
