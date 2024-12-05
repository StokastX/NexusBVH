#pragma once

#include <iostream>

namespace NXB
{
	uint32_t DivideRoundUp(uint32_t x, uint32_t y)
	{
		return 1 + ((x - 1) / y);
	}
}