#include "DeviceTraversal.h"

#include "NXB/BVHTraversal.h"
#include "NXB/DeviceBuffer.h"
#include "NXB/Error.h"

#include "PrimIntersect.h"

namespace NXB::Test
{
	namespace
	{
		constexpr uint32_t blockSize = 128;

		template <typename PrimT>
		__global__ void ClosestHitKernel(BVH2::DeviceView bvh, const PrimT* prims,
			const Ray* rays, uint32_t rayCount, Hit* hits)
		{
			const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
			if (i >= rayCount)
				return;

			const Ray ray = rays[i];
			Hit hit{ RayMiss, InvalidIdx };
			float tMax = RayMiss;

			TraverseBVH2(bvh, ray.origin, ray.invDirection, 0.0f, tMax,
				[&](uint32_t primIdx, float& tMax)
				{
					float t;
					if (IntersectPrim(prims[primIdx], ray, 0.0f, tMax, t))
					{
						tMax = t;
						hit.t = t;
						hit.primIdx = primIdx;
					}
					return true;
				});

			hits[i] = hit;
		}

		template <typename PrimT>
		__global__ void AnyHitKernel(BVH2::DeviceView bvh, const PrimT* prims,
			const Ray* rays, uint32_t rayCount, uint8_t* anyHit)
		{
			const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
			if (i >= rayCount)
				return;

			const Ray ray = rays[i];
			bool found = false;
			float tMax = RayMiss;

			const bool ranToEnd = TraverseBVH2(bvh, ray.origin, ray.invDirection, 0.0f, tMax,
				[&](uint32_t primIdx, float& tMax)
				{
					float t;
					if (IntersectPrim(prims[primIdx], ray, 0.0f, tMax, t))
						found = true;
					return !found;
				});

			// Reported as the early out flag, not as `found`, so a callback return that
			// the traversal ignores shows up here
			anyHit[i] = ranToEnd ? 0u : 1u;
		}

		template <typename PrimT>
		__global__ void BruteForceKernel(const PrimT* prims, uint32_t primCount,
			const Ray* rays, uint32_t rayCount, Hit* hits)
		{
			const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
			if (i >= rayCount)
				return;

			hits[i] = BruteForce(prims, primCount, rays[i], 0.0f, RayMiss);
		}

		uint32_t GridSize(size_t rayCount)
		{
			return (uint32_t)((rayCount + blockSize - 1) / blockSize);
		}

		template <typename PrimT>
		std::vector<Hit> RunClosestHit(const BVH2::DeviceView& bvh, const PrimT* prims,
			const std::vector<Ray>& rays)
		{
			DeviceBuffer<Ray> deviceRays(rays);
			DeviceBuffer<Hit> deviceHits(rays.size());

			ClosestHitKernel<PrimT><<<GridSize(rays.size()), blockSize>>>(
				bvh, prims, deviceRays.Get(), (uint32_t)rays.size(), deviceHits.Get());

			NXB_CUDA_CHECK(cudaGetLastError());
			return deviceHits.ToHost();
		}
	}

	std::vector<Hit> DeviceClosestHits(const BVH2::DeviceView& bvh,
		const Triangle* devicePrims, const std::vector<Ray>& rays)
	{
		return RunClosestHit(bvh, devicePrims, rays);
	}

	std::vector<Hit> DeviceClosestHits(const BVH2::DeviceView& bvh,
		const AABB* devicePrims, const std::vector<Ray>& rays)
	{
		return RunClosestHit(bvh, devicePrims, rays);
	}

	std::vector<uint8_t> DeviceAnyHits(const BVH2::DeviceView& bvh,
		const Triangle* devicePrims, const std::vector<Ray>& rays)
	{
		DeviceBuffer<Ray> deviceRays(rays);
		DeviceBuffer<uint8_t> deviceAnyHit(rays.size());

		AnyHitKernel<Triangle><<<GridSize(rays.size()), blockSize>>>(
			bvh, devicePrims, deviceRays.Get(), (uint32_t)rays.size(), deviceAnyHit.Get());

		NXB_CUDA_CHECK(cudaGetLastError());
		return deviceAnyHit.ToHost();
	}

	std::vector<Hit> DeviceBruteForceHits(const Triangle* devicePrims, uint32_t primCount,
		const std::vector<Ray>& rays)
	{
		DeviceBuffer<Ray> deviceRays(rays);
		DeviceBuffer<Hit> deviceHits(rays.size());

		BruteForceKernel<Triangle><<<GridSize(rays.size()), blockSize>>>(
			devicePrims, primCount, deviceRays.Get(), (uint32_t)rays.size(), deviceHits.Get());

		NXB_CUDA_CHECK(cudaGetLastError());
		return deviceHits.ToHost();
	}
}
