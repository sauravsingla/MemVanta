#pragma once
// Assertions for the MemVanta test suite.
//
// These deliberately do NOT use <cassert>. Every CI workflow builds with
// -DCMAKE_BUILD_TYPE=Release, whose default flags include -DNDEBUG, which compiles
// assert() to nothing -- so assert-based checks silently stop verifying anything in
// exactly the configuration that gates merges and benchmark publication.
//
// CHECK evaluates its condition in every build type, reports the failing expression
// and source location, and records the failure. Tests end with `return
// memvanta_test::failures();` so a broken check produces a non-zero exit status that
// ctest observes.
//
// CHECK does not abort, so one run reports every failure rather than only the first.
// Use CHECK_FATAL where continuing after a failure would be unsafe (for example when
// later code would dereference something the check just proved invalid).
#include <cstdio>
#include <cstdlib>

namespace memvanta_test {

inline int& failure_count() { static int n = 0; return n; }
inline int failures() { return failure_count() ? 1 : 0; }

inline void record(const char* expr, const char* file, int line, const char* note) {
    ++failure_count();
    std::fprintf(stderr, "CHECK failed: %s\n  at %s:%d", expr, file, line);
    if (note && *note) std::fprintf(stderr, "\n  %s", note);
    std::fputc('\n', stderr);
}

} // namespace memvanta_test

// CHECK and CHECK_MSG contain no return statement, so they are usable inside void
// helper functions as well as in main().
#define CHECK(cond)                                                              \
    do { if (!(cond)) memvanta_test::record(#cond, __FILE__, __LINE__, ""); } while (0)

#define CHECK_MSG(cond, note)                                                    \
    do { if (!(cond)) memvanta_test::record(#cond, __FILE__, __LINE__, note); } while (0)

// Only for cases where continuing after the failure would be unsafe; returns from
// the enclosing function, so it must be used where returning 1 is valid.
#define CHECK_FATAL(cond)                                                        \
    do {                                                                         \
        if (!(cond)) {                                                           \
            memvanta_test::record(#cond, __FILE__, __LINE__, "");                \
            return 1;                                                            \
        }                                                                        \
    } while (0)
