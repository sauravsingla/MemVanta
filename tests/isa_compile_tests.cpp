#include "check.hpp"

#include <cstddef>

int main() {
#if defined(__aarch64__)
    CHECK(sizeof(void*) == 8);
#if defined(__ARM_NEON)
    CHECK(true);
#else
    CHECK_MSG(false, "AArch64 validation is expected to compile with ARM NEON enabled");
#endif
#else
    // Native non-ARM CI still builds this target so the test registration stays
    // uniform across platforms. The ARM64/QEMU lane exercises the NEON assertion.
    CHECK(sizeof(void*) >= 4);
#endif
    return memvanta_test::failures();
}
