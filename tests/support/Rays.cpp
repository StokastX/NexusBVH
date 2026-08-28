#include "Rays.h"

#include <random>

#include "NXB/BVHTraversal.h"

namespace NXB::Test
{
	namespace
	{
		float3 Lerp(const float3& a, const float3& b, float t)
		{
			return make_float3(a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t, a.z + (b.z - a.z) * t);
		}

		float3 RandomPointInside(const AABB& bounds, std::mt19937& gen)
		{
			std::uniform_real_distribution<float> unit(0.0f, 1.0f);
			return Lerp(bounds.bMin, bounds.bMax, unit(gen));
		}

		// Radius of a sphere that comfortably encloses the scene
		float EnclosingRadius(const AABB& bounds)
		{
			float dx = bounds.bMax.x - bounds.bMin.x;
			float dy = bounds.bMax.y - bounds.bMin.y;
			float dz = bounds.bMax.z - bounds.bMin.z;
			return sqrtf(dx * dx + dy * dy + dz * dz);
		}
	}

	Ray MakeRay(float3 origin, float3 direction)
	{
		Ray ray;
		ray.origin = origin;
		ray.direction = direction;
		ray.invDirection = make_float3(1.0f / direction.x, 1.0f / direction.y, 1.0f / direction.z);
		return ray;
	}

	bool operator==(const Hit& a, const Hit& b)
	{
		return a.t == b.t && a.primIdx == b.primIdx;
	}

	std::vector<Ray> GenerateRays(const AABB& sceneBounds, uint32_t count)
	{
		std::vector<Ray> rays;
		rays.reserve(count);

		std::mt19937 gen(4242);
		std::uniform_real_distribution<float> unit(0.0f, 1.0f);

		const float3 centre = sceneBounds.Centroid();
		const float radius = EnclosingRadius(sceneBounds);

		for (uint32_t i = 0; i < count; ++i)
		{
			// Uniform on the sphere
			float z = 1.0f - 2.0f * unit(gen);
			float r = sqrtf(fmaxf(0.0f, 1.0f - z * z));
			float phi = 6.28318530718f * unit(gen);
			float3 origin = make_float3(
				centre.x + radius * r * cosf(phi),
				centre.y + radius * r * sinf(phi),
				centre.z + radius * z);

			float3 target = RandomPointInside(sceneBounds, gen);
			rays.push_back(MakeRay(origin, make_float3(
				target.x - origin.x, target.y - origin.y, target.z - origin.z)));
		}
		return rays;
	}

	std::vector<Ray> GenerateInteriorRays(const AABB& sceneBounds, uint32_t count)
	{
		std::vector<Ray> rays;
		rays.reserve(count);

		std::mt19937 gen(99991);
		std::uniform_real_distribution<float> sym(-1.0f, 1.0f);

		for (uint32_t i = 0; i < count; ++i)
		{
			float3 origin = RandomPointInside(sceneBounds, gen);
			float3 direction = make_float3(sym(gen), sym(gen), sym(gen));

			// A zero direction is not a ray; axis aligned cases are generated separately
			if (direction.x == 0.0f && direction.y == 0.0f && direction.z == 0.0f)
				direction.x = 1.0f;

			rays.push_back(MakeRay(origin, direction));
		}
		return rays;
	}

	std::vector<Ray> GenerateAxisAlignedRays(const AABB& sceneBounds, uint32_t perAxis)
	{
		std::vector<Ray> rays;
		rays.reserve(perAxis * 6);

		std::mt19937 gen(31337);
		const float radius = EnclosingRadius(sceneBounds);

		const float3 directions[6] = {
			make_float3(1.0f, 0.0f, 0.0f), make_float3(-1.0f, 0.0f, 0.0f),
			make_float3(0.0f, 1.0f, 0.0f), make_float3(0.0f, -1.0f, 0.0f),
			make_float3(0.0f, 0.0f, 1.0f), make_float3(0.0f, 0.0f, -1.0f)
		};

		for (uint32_t i = 0; i < perAxis; ++i)
		{
			float3 through = RandomPointInside(sceneBounds, gen);
			for (const float3& d : directions)
			{
				float3 origin = make_float3(
					through.x - d.x * radius, through.y - d.y * radius, through.z - d.z * radius);
				rays.push_back(MakeRay(origin, d));
			}
		}
		return rays;
	}

	std::vector<Ray> GenerateMissRays(const AABB& sceneBounds, uint32_t count)
	{
		std::vector<Ray> rays = GenerateRays(sceneBounds, count);

		// Same origins on the enclosing sphere, aimed directly away from the scene
		for (Ray& ray : rays)
			ray = MakeRay(ray.origin, make_float3(-ray.direction.x, -ray.direction.y, -ray.direction.z));

		return rays;
	}
}
