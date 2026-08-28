#pragma once

#include <cstdint>
#include <type_traits>
#include <utility>
#include <vector>

#include <vector_types.h>

#include "AABB.h"
#include "DeviceBuffer.h"

namespace NXB
{
	inline constexpr uint32_t InvalidIdx = ~0u;

	/* \brief Owns the device memory of a binary BVH
	 *
	 * Move only. The destructor releases the node array on the stream it was allocated on
	 * and back into the pool it came from, so neither has to be remembered by the caller.
	 *
	 * Pass View() into a kernel, ToHost() to read it back. Release() opts out and hands
	 * the raw arrays over, for a caller that would rather own them itself.
	 *
	 * The root is the LAST node, nodes[nodeCount - 1], because the builder merges bottom
	 * up and allocates the root last. BVH8 roots at node 0 instead, so a traversal written
	 * for one does not carry over to the other -- ask RootIdx() rather than assuming.
	 */
	class BVH2
	{
	public:
		struct Node
		{
			AABB bounds;

			// leftChild = InvalidIdx if leaf node
			uint32_t leftChild;

			// rightChild = primIdx if leaf node
			uint32_t rightChild;
		};

		/* \brief What a kernel receives
		 *
		 * Kernel parameters are bitwise copied into parameter space, so they have to be
		 * trivially copyable -- which an owning type is not. The pointer inside a view
		 * belongs to the BVH2 that handed it out and must not outlive it.
		 */
		struct DeviceView
		{
			Node* nodes;
			uint32_t nodeCount;
			uint32_t primCount;

			// Root bounds
			AABB bounds;

			// Index of the root node, InvalidIdx if the BVH is empty
			__host__ __device__ uint32_t RootIdx() const
			{
				return nodeCount ? nodeCount - 1 : InvalidIdx;
			}
		};

		// A host side copy that releases itself
		struct Host
		{
			std::vector<Node> nodes;
			uint32_t primCount = 0;

			// Root bounds
			AABB bounds;

			// Index of the root node, InvalidIdx if the BVH is empty
			uint32_t RootIdx() const
			{
				return nodes.empty() ? InvalidIdx : (uint32_t)nodes.size() - 1;
			}
		};

		BVH2() { m_bounds.Clear(); }

		BVH2(DeviceBuffer<Node>&& nodes, uint32_t primCount, const AABB& bounds)
			: m_nodes(std::move(nodes)), m_primCount(primCount), m_bounds(bounds) { }

		BVH2(const BVH2&) = delete;
		BVH2& operator=(const BVH2&) = delete;
		BVH2(BVH2&&) noexcept = default;
		BVH2& operator=(BVH2&&) noexcept = default;

		DeviceView View() const
		{
			return DeviceView{ m_nodes.Get(), (uint32_t)m_nodes.Count(), m_primCount, m_bounds };
		}

		Host ToHost() const
		{
			Host host;
			host.nodes = m_nodes.ToHost();
			host.primCount = m_primCount;
			host.bounds = m_bounds;
			return host;
		}

		// Gives up ownership. The caller is then responsible for the node array, which
		// came from the async allocator and has to go back to it with cudaFreeAsync.
		DeviceView Release()
		{
			DeviceView view = View();
			m_nodes.Release();
			m_primCount = 0;
			m_bounds.Clear();
			return view;
		}

		// Takes ownership of arrays this class did not allocate
		static BVH2 Adopt(const DeviceView& view, cudaStream_t stream = 0)
		{
			return BVH2(DeviceBuffer<Node>::Adopt(view.nodes, view.nodeCount, stream),
				view.primCount, view.bounds);
		}

		uint32_t NodeCount() const { return (uint32_t)m_nodes.Count(); }
		uint32_t PrimCount() const { return m_primCount; }

		// Index of the root node, InvalidIdx if the BVH is empty
		uint32_t RootIdx() const { return NodeCount() ? NodeCount() - 1 : InvalidIdx; }

		const AABB& Bounds() const { return m_bounds; }
		bool Empty() const { return m_nodes.Get() == nullptr; }

	private:
		DeviceBuffer<Node> m_nodes;
		uint32_t m_primCount = 0;
		AABB m_bounds;
	};

	static_assert(std::is_trivially_copyable<BVH2::DeviceView>::value,
		"BVH2::DeviceView is passed by value into kernels and must stay trivially copyable");

	/* \brief Owns the device memory of a compressed wide BVH (See Ylitie et al.)
	 *
	 * Same contract as BVH2, over two arrays. Note that the node array is allocated at the
	 * (4n - 1) / 7 worst case while NodeCount() is what the collapse actually produced, so
	 * the two differ -- ToHost copies only the nodes that exist.
	 *
	 * The root is node 0: the collapse walks top down and writes the root first, where the
	 * binary builder merges bottom up and writes it last. RootIdx() is the portable way to
	 * ask, and is the only reason the two agree on anything here.
	 */
	class BVH8
	{
	public:
		struct NodeExplicit
		{
			// Origin point of the local grid
			float3 p;

			// Scale of the grid
			uint8_t e[3];

			// 8-bit mask to indicate which of the children are internal nodes
			uint8_t imask = 0;

			// Index of the first child
			uint32_t childBaseIdx = 0;

			// Index of the first triangle
			uint32_t primBaseIdx = 0;

			// Field encoding the indexing information of every child
			uint8_t meta[8];

			// Quantized origin of the childs' AABBs
			uint8_t qlox[8], qloy[8], qloz[8];

			// Quantized end point of the childs' AABBs
			uint8_t qhix[8], qhiy[8], qhiz[8];

		};

		struct Node
		{
			// P (12 bytes), e (3 bytes), imask (1 byte)
			float4 p_e_imask;

			// Child base index (4 bytes), triangle base index (4 bytes), meta (8 bytes)
			float4 childidx_tridx_meta;

			// qlox (8 bytes), qloy (8 bytes)
			float4 qlox_qloy;

			// qloz (8 bytes), qlix (8 bytes)
			float4 qloz_qhix;

			// qliy (8 bytes), qliz (8 bytes)
			float4 qhiy_qhiz;
		};

		// See BVH2::DeviceView
		struct DeviceView
		{
			Node* nodes;
			uint32_t nodeCount;

			uint32_t* primIdx;
			uint32_t primCount;

			// Root bounds
			AABB bounds;

			// Index of the root node, InvalidIdx if the BVH is empty
			__host__ __device__ uint32_t RootIdx() const
			{
				return nodeCount ? 0 : InvalidIdx;
			}
		};

		// A host side copy that releases itself. nodes.size() is the node count.
		struct Host
		{
			std::vector<Node> nodes;
			std::vector<uint32_t> primIdx;
			uint32_t primCount = 0;

			// Root bounds
			AABB bounds;

			// Index of the root node, InvalidIdx if the BVH is empty
			uint32_t RootIdx() const { return nodes.empty() ? InvalidIdx : 0; }
		};

		BVH8() { m_bounds.Clear(); }

		BVH8(DeviceBuffer<Node>&& nodes, DeviceBuffer<uint32_t>&& primIdx,
			uint32_t nodeCount, uint32_t primCount, const AABB& bounds)
			: m_nodes(std::move(nodes)), m_primIdx(std::move(primIdx)),
			  m_nodeCount(nodeCount), m_primCount(primCount), m_bounds(bounds) { }

		BVH8(const BVH8&) = delete;
		BVH8& operator=(const BVH8&) = delete;
		BVH8(BVH8&&) noexcept = default;
		BVH8& operator=(BVH8&&) noexcept = default;

		DeviceView View() const
		{
			return DeviceView{ m_nodes.Get(), m_nodeCount, m_primIdx.Get(), m_primCount, m_bounds };
		}

		Host ToHost() const
		{
			Host host;
			host.nodes.resize(m_nodeCount);
			m_nodes.Download(host.nodes.data(), m_nodeCount);
			host.primIdx = m_primIdx.ToHost();
			host.primCount = m_primCount;
			host.bounds = m_bounds;
			return host;
		}

		// See BVH2::Release
		DeviceView Release()
		{
			DeviceView view = View();
			m_nodes.Release();
			m_primIdx.Release();
			m_nodeCount = 0;
			m_primCount = 0;
			m_bounds.Clear();
			return view;
		}

		/* \param allocatedNodeCount How many nodes the array behind view.nodes holds. It
		 *        is the (4n - 1) / 7 bound rather than view.nodeCount for an array this
		 *        library produced, and only matters to the free.
		 */
		static BVH8 Adopt(const DeviceView& view, size_t allocatedNodeCount, cudaStream_t stream = 0)
		{
			return BVH8(DeviceBuffer<Node>::Adopt(view.nodes, allocatedNodeCount, stream),
				DeviceBuffer<uint32_t>::Adopt(view.primIdx, view.primCount, stream),
				view.nodeCount, view.primCount, view.bounds);
		}

		uint32_t NodeCount() const { return m_nodeCount; }
		uint32_t PrimCount() const { return m_primCount; }

		// Index of the root node, InvalidIdx if the BVH is empty
		uint32_t RootIdx() const { return m_nodeCount ? 0 : InvalidIdx; }

		/*
		 * Average number of children per node, a coarse measure of how well the collapse
		 * filled its 8 slots. Pure host arithmetic on the two counts above, so it costs
		 * nothing to ask for.
		 *
		 * Warning: the formula is only valid while a leaf holds exactly one primitive.
		 * Should be (totalNodes - 1) / internalNodes.
		 */
		float AverageChildPerNode() const
		{
			return m_nodeCount ? (float)(m_primCount + m_nodeCount - 1) / m_nodeCount : 0.0f;
		}

		const AABB& Bounds() const { return m_bounds; }
		bool Empty() const { return m_nodes.Get() == nullptr; }

	private:
		DeviceBuffer<Node> m_nodes;
		DeviceBuffer<uint32_t> m_primIdx;
		uint32_t m_nodeCount = 0;
		uint32_t m_primCount = 0;
		AABB m_bounds;
	};

	static_assert(std::is_trivially_copyable<BVH8::DeviceView>::value,
		"BVH8::DeviceView is passed by value into kernels and must stay trivially copyable");
}
