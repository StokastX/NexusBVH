#pragma once

#include <iostream>

namespace NXB
{
	// Must be inline: this is a free function in a header, and it links today only
	// because exactly one translation unit includes it.
	inline constexpr uint32_t DivideRoundUp(uint32_t x, uint32_t y)
	{
		// x == 0 would underflow to 0xFFFFFFFF / y
		return x == 0 ? 0 : 1 + ((x - 1) / y);
	}
}