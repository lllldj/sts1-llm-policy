#pragma once

// sts_lightspeed currently relies on transitive standard-library includes for
// std::find, std::find_if, std::sort, and std::stable_sort. Inject this header
// at compile time so the pinned upstream checkout can remain unmodified.
#include <algorithm>
