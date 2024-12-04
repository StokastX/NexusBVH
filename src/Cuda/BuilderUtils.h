#pragma once

#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "Math/CudaMath.h"
#include "Math/AABB.h"

namespace NXB
{
	// Float version of atomicMin
	__device__ __forceinline__ void atomicMin(float* ptr, float value)
	{
		unsigned int curr = atomicAdd((unsigned int*)ptr, 0);
		while (value < __int_as_float(curr)) {
			unsigned int prev = curr;
			curr = atomicCAS((unsigned int*)ptr, curr, __float_as_int(value));
			if (curr == prev)
				break;
		}
	}

	// Float version of atomicMax
	__device__ __forceinline__ void atomicMax(float* ptr, float value)
	{
		unsigned int curr = atomicAdd((unsigned int*)ptr, 0);
		while (value > __int_as_float(curr)) {
			unsigned int prev = curr;
			curr = atomicCAS((unsigned int*)ptr, curr, __float_as_int(value));
			if (curr == prev)
				break;
		}
	}

	__device__ void AtomicGrow(AABB* aabb, const AABB& other)
	{
		atomicMin(&aabb->bMin.x, other.bMin.x);
		atomicMin(&aabb->bMin.y, other.bMin.y);
		atomicMin(&aabb->bMin.z, other.bMin.z);

		atomicMax(&aabb->bMax.x, other.bMax.x);
		atomicMax(&aabb->bMax.y, other.bMax.y);
		atomicMax(&aabb->bMax.z, other.bMax.z);
	}

	/* \brief Interleave the first 21 bits of x every three bits,
	 * ie insert two zeroes between every of the first 21 bits of x
	 * \param x Quantitized position, must be between 0 and 2^21 - 1 = 2,097,152
	 */
	__device__ uint64_t InterleaveBits(uint64_t x)
	{
		/* Comments generated with Python from https://stackoverflow.com/questions/18529057/produce-interleaving-bit-patterns-morton-keys-for-32-bit-64-bit-and-128bit */

		/*
		 * Current Mask:           0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0001 1111 1111 1111 1111 1111
		 * Which bits to shift:    0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0001 1111 0000 0000 0000 0000  hex: 0x1f0000
		 * Shifted part (<< 32):   0000 0000 0001 1111 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000  hex: 0x1f000000000000
		 * NonShifted Part:        0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 1111 1111 1111 1111  hex: 0xffff
		 * Bitmask is now :        0000 0000 0001 1111 0000 0000 0000 0000 0000 0000 0000 0000 1111 1111 1111 1111  hex: 0x1f00000000ffff
		 */
		x = (x | (x << 32)) & 0x1f00000000ffff;

		/* 
		 * Current Mask:           0000 0000 0001 1111 0000 0000 0000 0000 0000 0000 0000 0000 1111 1111 1111 1111
		 * Which bits to shift:    0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 1111 1111 0000 0000  hex: 0xff00
		 * Shifted part (<< 16):   0000 0000 0000 0000 0000 0000 0000 0000 1111 1111 0000 0000 0000 0000 0000 0000  hex: 0xff000000
		 * NonShifted Part:        0000 0000 0001 1111 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 1111 1111  hex: 0x1f0000000000ff
		 * Bitmask is now :        0000 0000 0001 1111 0000 0000 0000 0000 1111 1111 0000 0000 0000 0000 1111 1111  hex: 0x1f0000ff0000ff
		 */
		x = (x | (x << 16)) & 0x1f0000ff0000ff;

		/* 
		 * Current Mask:           0000 0000 0001 1111 0000 0000 0000 0000 1111 1111 0000 0000 0000 0000 1111 1111
		 * Which bits to shift:    0000 0000 0001 0000 0000 0000 0000 0000 1111 0000 0000 0000 0000 0000 1111 0000  hex: 0x100000f00000f0
		 * Shifted part (<< 8):    0001 0000 0000 0000 0000 0000 1111 0000 0000 0000 0000 0000 1111 0000 0000 0000  hex: 0x100000f00000f000
		 * NonShifted Part:        0000 0000 0000 1111 0000 0000 0000 0000 0000 1111 0000 0000 0000 0000 0000 1111  hex: 0xf00000f00000f
		 * Bitmask is now :        0001 0000 0000 1111 0000 0000 1111 0000 0000 1111 0000 0000 1111 0000 0000 1111  hex: 0x100f00f00f00f00f
		 */
		x = (x | (x << 8)) & 0x100f00f00f00f00f;

		/* 
		 * Current Mask:           0001 0000 0000 1111 0000 0000 1111 0000 0000 1111 0000 0000 1111 0000 0000 1111
		 * Which bits to shift:    0000 0000 0000 1100 0000 0000 1100 0000 0000 1100 0000 0000 1100 0000 0000 1100  hex: 0xc00c00c00c00c
		 * Shifted part (<< 4):    0000 0000 1100 0000 0000 1100 0000 0000 1100 0000 0000 1100 0000 0000 1100 0000  hex: 0xc00c00c00c00c0
		 * NonShifted Part:        0001 0000 0000 0011 0000 0000 0011 0000 0000 0011 0000 0000 0011 0000 0000 0011  hex: 0x1003003003003003
		 * Bitmask is now :        0001 0000 1100 0011 0000 1100 0011 0000 1100 0011 0000 1100 0011 0000 1100 0011  hex: 0x10c30c30c30c30c3
		 */
		x = (x | (x << 4)) & 0x10c30c30c30c30c3;

		/* 
		 * Current Mask:           0001 0000 1100 0011 0000 1100 0011 0000 1100 0011 0000 1100 0011 0000 1100 0011
		 * Which bits to shift:    0000 0000 1000 0010 0000 1000 0010 0000 1000 0010 0000 1000 0010 0000 1000 0010  hex: 0x82082082082082
		 * Shifted part (<< 2):    0000 0010 0000 1000 0010 0000 1000 0010 0000 1000 0010 0000 1000 0010 0000 1000  hex: 0x208208208208208
		 * NonShifted Part:        0001 0000 0100 0001 0000 0100 0001 0000 0100 0001 0000 0100 0001 0000 0100 0001  hex: 0x1041041041041041
		 * Bitmask is now :        0001 0010 0100 1001 0010 0100 1001 0010 0100 1001 0010 0100 1001 0010 0100 1001  hex: 0x1249249249249249
		*/
		x = (x | (x << 2)) & 0x1249249249249249;

		return x;
	}

	/* \brief Compute a 64-bit Morton code for the given quantitized 3D point
	 * \param x The quantitized x coordinate
	 * \param y The quantitized y coordinate
	 * \param z The quantitized z coordinate
	 */
	__device__ uint64_t MortonCode(uint32_t x, uint32_t y, uint32_t z)
	{
		return InterleaveBits(x) | InterleaveBits(y) << 1 | InterleaveBits(z) << 2;
	}

	/* \brief Compute a 64-bit Morton code for the given (not normalized) 3D point
	 * \param centroid The centroid position, normalized in [0, 1]
	 */
	__device__ uint64_t MortonCode(const float3& centroid)
	{
		uint32_t x = centroid.x * 0x1fffff;
		uint32_t y = centroid.y * 0x1fffff;
		uint32_t z = centroid.z * 0x1fffff;
		return MortonCode(x, y, z);
	}
}