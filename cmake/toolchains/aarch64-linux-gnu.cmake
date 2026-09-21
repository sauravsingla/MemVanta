set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

set(CMAKE_C_COMPILER aarch64-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++)
set(CMAKE_FIND_ROOT_PATH /usr/aarch64-linux-gnu)

set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

# CTest automatically prefixes cross-compiled test executables with this
# emulator, allowing portable correctness checks to run on hosted x86 runners.
set(CMAKE_CROSSCOMPILING_EMULATOR
    qemu-aarch64;-L;/usr/aarch64-linux-gnu
    CACHE STRING "ARM64 user-mode emulator" FORCE)
