cmake_minimum_required(VERSION 3.16)

project(qtrobot_yolo_driver
    VERSION 0.1.0
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

include(GNUInstallDirs)
include(FetchContent)

find_package(Threads REQUIRED)
find_package(paramify REQUIRED)
find_package(magpie REQUIRED)
find_package(OpenCV REQUIRED)

# =========================
# YOLOs-CPP (vendored — header-only, no usable CMake target of its own)
# =========================
#
# YOLOs-CPP's own CMakeLists.txt builds its example executables and requires
# ONNXRUNTIME_DIR to already point at an extracted prebuilt ONNX Runtime
# release — it isn't designed to be add_subdirectory()'d by a consumer.
# We only need its headers, so fetch the source without configuring its
# build (same FetchContent_Populate-only pattern used for other vendored
# header-heavy dependencies in this project family).
FetchContent_Declare(yolos_cpp
    GIT_REPOSITORY https://github.com/Geekgineer/YOLOs-CPP.git
    GIT_TAG        v1.0.0
)
FetchContent_GetProperties(yolos_cpp)
if(NOT yolos_cpp_POPULATED)
    FetchContent_Populate(yolos_cpp)
endif()

# =========================
# ONNX Runtime (prebuilt release tarball, not built from source)
# =========================
#
# Single CPU-only build on both platforms for now (per project decision —
# YOLOs-CPP's own session_base.hpp probes Ort::GetAvailableProviders() at
# runtime and falls back to CPU automatically if CUDA EP isn't present in
# the linked build, so this is safe even when device=cuda is requested).
# A GPU-capable ONNX Runtime build for Jetson is a later, separate exercise
# (Microsoft doesn't publish one for linux-aarch64 — needs NVIDIA's own
# Jetson build or building from source against JetPack's CUDA/cuDNN).
set(ONNXRUNTIME_VERSION "1.20.1")
if(CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
    set(_ort_arch "x64")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    set(_ort_arch "aarch64")
else()
    message(FATAL_ERROR "Unsupported architecture for ONNX Runtime: ${CMAKE_SYSTEM_PROCESSOR}")
endif()

FetchContent_Declare(onnxruntime
    URL https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-linux-${_ort_arch}-${ONNXRUNTIME_VERSION}.tgz
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_MakeAvailable(onnxruntime)
set(ONNXRUNTIME_DIR ${onnxruntime_SOURCE_DIR})
set(ONNXRUNTIME_LIB ${ONNXRUNTIME_DIR}/lib/libonnxruntime.so)

# =========================
# motcpp (ByteTrack) — proper CMake package, FetchContent_MakeAvailable works directly
# =========================

set(MOTCPP_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(MOTCPP_BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
set(MOTCPP_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(MOTCPP_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
# We only use ByteTrack (motion-only); ReID backend needs its own ONNX Runtime
# setup we don't need — keep it off to avoid a second, redundant ORT fetch.
set(MOTCPP_ENABLE_ONNX OFF CACHE BOOL "" FORCE)
set(MOTCPP_INSTALL OFF CACHE BOOL "" FORCE)

# No real release tag exists in this repo (its own README's documented
# "GIT_TAG v1.0.0" example doesn't match an actual tag — verified via
# `git ls-remote --tags`, only data/model tags exist, not source releases).
# Pin to main's current HEAD commit instead, for reproducibility.
FetchContent_Declare(motcpp
    GIT_REPOSITORY https://github.com/Geekgineer/motcpp.git
    GIT_TAG        b35c1897b15ae8ea1bee3d91d31165b08550fe8a
)
FetchContent_MakeAvailable(motcpp)

# =========================
# Executable
# =========================

add_executable(qtrobot_yolo_driver
    src/utility.cpp
    src/fast_pose_detector.cpp
    src/pose_backend.cpp
    src/yolo_driver_node.cpp
)

set(LUXAI_PREFIX /opt/luxai/${PROJECT_NAME})

target_include_directories(qtrobot_yolo_driver
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${yolos_cpp_SOURCE_DIR}/include
        ${ONNXRUNTIME_DIR}/include
)

target_link_libraries(qtrobot_yolo_driver
    PUBLIC
        Threads::Threads
    PRIVATE
        magpie::magpie
        paramify::paramify
        motcpp::motcpp
        ${OpenCV_LIBS}
        ${ONNXRUNTIME_LIB}
)

# ONNX Runtime ships as a shared library with no soname-based discovery —
# same RPATH approach YOLOs-CPP's own build uses, so the binary finds
# libonnxruntime.so without needing LD_LIBRARY_PATH set at runtime.
# BUILD_RPATH points at the FetchContent-fetched copy (for running directly
# out of the build tree); INSTALL_RPATH must point at where install() below
# actually places the bundled .so on the target machine, not the build tree.
set_target_properties(qtrobot_yolo_driver PROPERTIES
    BUILD_RPATH   "${ONNXRUNTIME_DIR}/lib"
    INSTALL_RPATH "${LUXAI_PREFIX}/lib"
)

# =========================
# Install rules
# =========================

install(TARGETS qtrobot_yolo_driver
    RUNTIME DESTINATION ${LUXAI_PREFIX}/bin
)

install(FILES
    config/config.yaml
    DESTINATION ${LUXAI_PREFIX}/etc
)

install(DIRECTORY
    models/
    DESTINATION ${LUXAI_PREFIX}/models
)

# install(FILES) on a single symlink (libonnxruntime.so) only copies the
# symlink itself, not the real file it (transitively) points to —
# libonnxruntime.so -> libonnxruntime.so.1 -> libonnxruntime.so.1.20.1.
# install(DIRECTORY) preserves the whole symlink chain correctly.
install(DIRECTORY
    ${ONNXRUNTIME_DIR}/lib/
    DESTINATION ${LUXAI_PREFIX}/lib
    FILES_MATCHING PATTERN "*.so*"
)

install(FILES
    packaging/qtrobot-yolo-driver.service
    DESTINATION /lib/systemd/system
)

# =========================
# CPack (Debian)
# =========================

set(CPACK_GENERATOR "DEB")

set(CPACK_PACKAGE_NAME ${PROJECT_NAME})
set(CPACK_PACKAGE_VERSION ${PROJECT_VERSION})
set(CPACK_PACKAGE_VENDOR "LuxAI S.A.")
set(CPACK_PACKAGE_CONTACT "LuxAI <support@luxai.com>")

set(CPACK_DEBIAN_PACKAGE_MAINTAINER "LuxAI")

# -------------------------
# Architecture (Debian-compliant)
# -------------------------
if(CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
    set(CPACK_DEBIAN_PACKAGE_ARCHITECTURE "amd64")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    set(CPACK_DEBIAN_PACKAGE_ARCHITECTURE "arm64")
else()
    set(CPACK_DEBIAN_PACKAGE_ARCHITECTURE ${CMAKE_SYSTEM_PROCESSOR})
endif()

# -------------------------
# Ubuntu release (dynamic)
# -------------------------
execute_process(
    COMMAND lsb_release -rs
    OUTPUT_VARIABLE LINUX_VERSION
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
)
if(LINUX_VERSION)
    set(CPACK_DEBIAN_PACKAGE_RELEASE "1deb${LINUX_VERSION}")
else()
    set(CPACK_DEBIAN_PACKAGE_RELEASE "1")
endif()

# -------------------------
# Runtime dependencies
# -------------------------
# libonnxruntime.so ships inside the package itself (installed to
# ${LUXAI_PREFIX}/lib, see install() above + RPATH) since it's not a
# regular apt package on either platform.
#
# OpenCV's runtime package name varies across Ubuntu releases
# (libopencv-core4.5d, libopencv-core406, ...) — rather than guess, let
# dpkg-shlibdeps inspect the actual binary and resolve the right package(s)
# automatically. dpkg-shlibdeps would otherwise also try (and fail) to find
# a system package providing our bundled libonnxruntime.so, since no apt
# package provides it — CPACK_DEBIAN_PACKAGE_SHLIBDEPS_PRIVATE_DIRS tells it
# that directory's libraries are private/bundled, not a missing dependency.
set(CPACK_DEBIAN_PACKAGE_SHLIBDEPS ON)
set(CPACK_DEBIAN_PACKAGE_SHLIBDEPS_PRIVATE_DIRS "${LUXAI_PREFIX}/lib")
set(CPACK_DEBIAN_PACKAGE_DEPENDS
    "libmagpie (>= 0.7),
     libyaml-cpp0.6 | libyaml-cpp0.7 | libyaml-cpp0.8,
     libzmq5 | libzmq5t64"
)

# -------------------------
# Maintainer scripts
# -------------------------
set(CPACK_DEBIAN_PACKAGE_CONTROL_EXTRA
    ${CMAKE_CURRENT_SOURCE_DIR}/packaging/postinst
    ${CMAKE_CURRENT_SOURCE_DIR}/packaging/prerm
)

set(CPACK_DEBIAN_FILE_NAME "DEB-DEFAULT")

include(CPack)
