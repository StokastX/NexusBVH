#pragma once

#include <string>

#include "vendor/doctest.h"

#include "support/BVHChecks.h"

/*
 * The bridge between the framework-free checks in support/ and doctest.
 *
 * It lives here rather than in support/ on purpose: keeping the checks themselves free
 * of any framework is what lets them be called from somewhere that is not a test.
 */
namespace NXB::Test
{
	// Reports every error the checks collected as its own failed assertion, so the
	// output names what actually broke rather than just which case broke
	inline void CheckValid(const ValidationResult& result)
	{
		for (const std::string& error : result.errors)
			FAIL_CHECK(error);
	}
}
