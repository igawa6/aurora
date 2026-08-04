// Globals the converter reaches for that normally come from the live renderer,
// following the same pattern as os_test_globals.cpp and gx_test_stubs.cpp.
//
// The capability flags stay FALSE on purpose: false is what every Android GPU
// reports for BC, and the CPU fallback these tests cover is reached only when
// the GPU cannot sample the format itself.

#include <aurora/aurora.h>

namespace aurora {
AuroraConfig g_config{};

namespace webgpu {
bool g_bcTexturesSupported = false;
bool g_astcTexturesSupported = false;
bool g_textureComponentSwizzleSupported = false;
}  // namespace webgpu
}  // namespace aurora
