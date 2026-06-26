cmake_minimum_required(VERSION 3.18)

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
# Backend selection: onnx (ONNX Runtime, CPU or CUDA EP) vs tensorrt (TensorRT,
# GPU only). Defaults by platform — TensorRT/CUDA Toolkit ship as part of
# JetPack on Jetson, so there's no equivalent build-from-source/extraction
# friction there the way there is fetching a CUDA-EP-enabled ONNX Runtime for
# aarch64. x86 has no such constraint either way, so it defaults to the
# simpler ONNX/CPU path.
# =========================
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    set(_yolo_backend_default "tensorrt")
else()
    set(_yolo_backend_default "onnx")
endif()
set(YOLO_DRIVER_BACKEND ${_yolo_backend_default} CACHE STRING "Pose inference backend: onnx or tensorrt")
set_property(CACHE YOLO_DRIVER_BACKEND PROPERTY STRINGS "onnx" "tensorrt")
message(STATUS "qtrobot-yolo-driver backend: ${YOLO_DRIVER_BACKEND}")

# =========================
# ONNX Runtime (prebuilt release tarball, not built from source)
# =========================
#
# Unconditional regardless of YOLO_DRIVER_BACKEND: the "onnx" pose backend
# uses it directly, and the "tensorrt" pose backend still needs it for
# emotion recognition (EmotiEffLib has no TensorRT backend, only torch/onnx).
#
# Single CPU-only build on both platforms for now (per project decision —
# YOLOs-CPP's own session_base.hpp probes Ort::GetAvailableProviders() at
# runtime and falls back to CPU automatically if CUDA EP isn't present in
# the linked build, so this is safe even when device=cuda is requested).
# A GPU-capable ONNX Runtime build for Jetson is deliberately not pursued —
# see the "tensorrt" pose backend instead, which uses what JetPack already ships.
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

# Upstream packaging bug in the official ONNX Runtime v1.20.1 Linux release
# tarball: its own bundled CMake package
# (lib/cmake/onnxruntime/onnxruntimeTargets-release.cmake) hardcodes the
# library at "lib64/libonnxruntime.so.1.20.1", but the tarball actually
# ships everything under "lib/". EmotiEffLib's CMakeLists.txt below calls
# find_package(onnxruntime REQUIRED), which fails outright on this mismatch.
# Not a patch to any vendored/third-party file — just aliasing the path
# their generated config expects to the files that are actually here.
if(NOT EXISTS ${ONNXRUNTIME_DIR}/lib64)
    file(CREATE_LINK ${ONNXRUNTIME_DIR}/lib ${ONNXRUNTIME_DIR}/lib64 SYMBOLIC)
endif()

# =========================
# EmotiEffLib (emotion recognition, optional at runtime via emotion.enabled —
# always compiled in, always uses the ONNX Runtime above regardless of pose
# backend). Apache-2.0, proper CMake library target (unlike YOLOs-CPP), so a
# normal FetchContent_MakeAvailable + add_subdirectory works directly.
# =========================
set(BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(WITH_ONNX ${ONNXRUNTIME_DIR} CACHE PATH "" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)

FetchContent_Declare(emotiefflib
    GIT_REPOSITORY https://github.com/sb-ai-lab/EmotiEffLib.git
    # Not the v1.1.1 tag: its facial_analysis.cpp includes both the torch
    # and onnx backend headers unconditionally, so it fails to build
    # whenever only one of WITH_TORCH/WITH_ONNX is enabled (our case — onnx
    # only). Not main HEAD either: a later commit bumps the vendored
    # xtl/xtensor submodules to versions requiring CMake 3.29. Pinned to the
    # commit right after v1.1.1 that fixes the include guards but predates
    # that submodule bump.
    GIT_TAG        2f1bf61
    GIT_SUBMODULES_RECURSE TRUE
    SOURCE_SUBDIR  emotieffcpplib
)
FetchContent_MakeAvailable(emotiefflib)

# At this pinned commit, EmotiEffLib's own facial_analysis.cpp is
# internally inconsistent: getAvailableBackends()/checkBackend() check
# #ifdef WITH_ONNX (which we do define, above), but createInstance() checks
# #elif defined(USE_ONNXRUNTIME) instead — a different macro name, never
# defined by their own CMakeLists.txt under any option. Without this,
# checkBackend("onnx") succeeds but createInstance() then always falls
# through to "wasn't compiled with any backend support." Defining it
# ourselves for their target is configuration, not a patch to their files.
target_compile_definitions(emotiefflib PRIVATE USE_ONNXRUNTIME)

if(YOLO_DRIVER_BACKEND STREQUAL "onnx")

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

    set(_pose_backend_sources src/fast_pose_detector.cpp src/pose_backend_onnx.cpp)
    set(_pose_backend_include_dirs ${yolos_cpp_SOURCE_DIR}/include ${ONNXRUNTIME_DIR}/include)
    set(_pose_backend_link_libs ${ONNXRUNTIME_LIB})

else() # tensorrt

    # =========================
    # TensorRT + CUDA Toolkit — both ship as part of JetPack on Jetson, found
    # as ordinary system libraries (same search-path pattern YOLOs-CPP-TensorRT's
    # own CMakeLists uses). On x86 these would need a manual TensorRT install.
    # =========================
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)

    find_package(CUDAToolkit REQUIRED)
    message(STATUS "CUDA Toolkit: ${CUDAToolkit_VERSION}")

    set(_is_jetson OFF)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64" AND
       (EXISTS "/etc/nv_tegra_release" OR EXISTS "/usr/lib/aarch64-linux-gnu/tegra"))
        set(_is_jetson ON)
    endif()
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        if(_is_jetson)
            set(CMAKE_CUDA_ARCHITECTURES "72;87") # Xavier=72, Orin=87
        else()
            set(CMAKE_CUDA_ARCHITECTURES "75;80;86;89;90")
        endif()
    endif()
    message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")

    if(NOT DEFINED TENSORRT_DIR)
        set(_tensorrt_search_paths
            "/usr/lib/x86_64-linux-gnu"
            "/usr/lib/aarch64-linux-gnu"
            "/usr/local/tensorrt"
            "$ENV{TENSORRT_DIR}"
        )
        foreach(_path ${_tensorrt_search_paths})
            if(EXISTS "${_path}/libnvinfer.so" OR EXISTS "${_path}/lib/libnvinfer.so")
                set(TENSORRT_DIR "${_path}")
                break()
            endif()
        endforeach()
    endif()

    find_path(TENSORRT_INCLUDE_DIR NvInfer.h
        HINTS ${TENSORRT_DIR}/include /usr/include/x86_64-linux-gnu /usr/include/aarch64-linux-gnu /usr/include
    )
    find_library(NVINFER_LIB nvinfer
        HINTS ${TENSORRT_DIR}/lib ${TENSORRT_DIR} /usr/lib/x86_64-linux-gnu /usr/lib/aarch64-linux-gnu
    )
    find_library(NVONNXPARSER_LIB nvonnxparser
        HINTS ${TENSORRT_DIR}/lib ${TENSORRT_DIR} /usr/lib/x86_64-linux-gnu /usr/lib/aarch64-linux-gnu
    )

    if(NOT NVINFER_LIB)
        message(FATAL_ERROR
            "TensorRT not found!\n"
            "For Jetson, TensorRT is typically pre-installed via JetPack.\n"
            "For x86, install it or set -DTENSORRT_DIR=/path/to/tensorrt."
        )
    endif()
    message(STATUS "TensorRT include: ${TENSORRT_INCLUDE_DIR}")
    message(STATUS "TensorRT nvinfer: ${NVINFER_LIB}")

    # =========================
    # YOLOs-CPP-TensorRT (vendored — same FetchContent_Populate-only pattern as
    # the onnx backend's YOLOs-CPP; its own CMakeLists builds example
    # executables we don't need). No real release tag in this repo either
    # (verified via git ls-remote --tags — none exist) — pin to main's HEAD.
    # =========================
    FetchContent_Declare(yolos_trt
        GIT_REPOSITORY https://github.com/Geekgineer/YOLOs-CPP-TensorRT.git
        GIT_TAG        3617f2fa48f086440949d5164da76effd7bcef41
    )
    FetchContent_GetProperties(yolos_trt)
    if(NOT yolos_trt_POPULATED)
        FetchContent_Populate(yolos_trt)
    endif()

    # The one piece of YOLOs-CPP-TensorRT that isn't header-only: a small CUDA
    # kernel for GPU-side letterbox/preprocessing. Compiled the same way their
    # own CMakeLists does it.
    add_library(yolos_trt_cuda_preprocessing STATIC
        ${yolos_trt_SOURCE_DIR}/include/yolos/core/cuda_preprocessing.cu
    )
    target_include_directories(yolos_trt_cuda_preprocessing PRIVATE
        ${yolos_trt_SOURCE_DIR}/include
        ${TENSORRT_INCLUDE_DIR}
    )
    target_link_libraries(yolos_trt_cuda_preprocessing PRIVATE CUDA::cudart)
    set_target_properties(yolos_trt_cuda_preprocessing PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        POSITION_INDEPENDENT_CODE ON
    )

    set(_pose_backend_sources src/pose_backend_tensorrt.cpp)
    set(_pose_backend_include_dirs ${yolos_trt_SOURCE_DIR}/include ${TENSORRT_INCLUDE_DIR})
    set(_pose_backend_link_libs ${NVINFER_LIB} CUDA::cudart yolos_trt_cuda_preprocessing)
    if(NVONNXPARSER_LIB)
        list(APPEND _pose_backend_link_libs ${NVONNXPARSER_LIB})
    endif()

endif()

# =========================
# motcpp (ByteTrack) — backend-independent, needed by both
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
    src/yolo_driver_node.cpp
    src/emotion_recognizer.cpp
    ${_pose_backend_sources}
)

set(LUXAI_PREFIX /opt/luxai/${PROJECT_NAME})

target_include_directories(qtrobot_yolo_driver
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${_pose_backend_include_dirs}
)

target_link_libraries(qtrobot_yolo_driver
    PUBLIC
        Threads::Threads
    PRIVATE
        magpie::magpie
        paramify::paramify
        motcpp::motcpp
        emotiefflib
        ${OpenCV_LIBS}
        ${_pose_backend_link_libs}
)

target_include_directories(qtrobot_yolo_driver
    PRIVATE
        ${emotiefflib_SOURCE_DIR}/emotieffcpplib/include
)

# ONNX Runtime ships as a shared library with no soname-based discovery — same
# RPATH approach YOLOs-CPP's own build uses, so the binary finds
# libonnxruntime.so without needing LD_LIBRARY_PATH set at runtime. Needed
# regardless of YOLO_DRIVER_BACKEND now, since emotiefflib always links it.
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

# Ship exactly two model files, named explicitly (no blanket extension
# globs): the pose model in whichever format the compiled-in backend loads,
# plus the emotion model, which is always ONNX regardless of pose backend.
if(YOLO_DRIVER_BACKEND STREQUAL "onnx")
    set(_pose_model_ext ".onnx")
else()
    set(_pose_model_ext ".trt")
endif()
install(FILES
    models/yolo26n-pose${_pose_model_ext}
    models/enet_b0_8_best_vgaf.onnx
    DESTINATION ${LUXAI_PREFIX}/models
)

# install(FILES) on a single symlink (libonnxruntime.so) only copies the
# symlink itself, not the real file it (transitively) points to —
# libonnxruntime.so -> libonnxruntime.so.1 -> libonnxruntime.so.1.20.1.
# install(DIRECTORY) preserves the whole symlink chain correctly. Needed
# regardless of YOLO_DRIVER_BACKEND now, since emotiefflib always links it.
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
# OpenCV's runtime package name varies across Ubuntu releases
# (libopencv-core4.5d, libopencv-core406, ...) — rather than guess, let
# dpkg-shlibdeps inspect the actual binary and resolve the right package(s)
# automatically.
set(CPACK_DEBIAN_PACKAGE_SHLIBDEPS ON)
# libonnxruntime.so ships inside the package itself (installed to
# ${LUXAI_PREFIX}/lib, see install() above + RPATH) since it's not a regular
# apt package on either platform — needed unconditionally now since
# emotiefflib always links ONNX Runtime regardless of pose backend.
# dpkg-shlibdeps would otherwise also try (and fail) to find a system
# package providing it — CPACK_DEBIAN_PACKAGE_SHLIBDEPS_PRIVATE_DIRS tells it
# that directory's libraries are private/bundled, not a missing dependency.
set(CPACK_DEBIAN_PACKAGE_SHLIBDEPS_PRIVATE_DIRS "${LUXAI_PREFIX}/lib")
# tensorrt backend's pose path needs no additional private-dirs entry:
# nvinfer/cudart are regular system libraries on Jetson (from JetPack), so
# dpkg-shlibdeps resolves them as ordinary (if unversioned-package-name)
# dependencies.
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
