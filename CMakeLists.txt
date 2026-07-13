cmake_minimum_required(VERSION 3.28)
project(pacd
    VERSION 0.1.0
    DESCRIPTION "Primitive-approximate convex decomposition of triangle meshes into collision proxies"
)
set(CMAKE_CXX_SCAN_FOR_MODULES OFF) # Causes errors with clang-tidy when used with gcc builds.

# Include CMake modules for target helpers and compiler settings
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(TargetHelpers)
include(GNUInstallDirs)

# ---- Build options ----------------------------------------------------------
# The core solver library is dependency-light (only Threads) and builds as C++20.
# Everything heavy -- the interactive viewer, the software renderer, dev tools,
# tests, and benchmarks -- pulls in the GL + logging stack (glfw/glad/imgui/OpenGL/
# spdlog/fmt) and is gated behind these options. They default ON only when pacd is
# the top-level project, so a downstream add_subdirectory / FetchContent / find_package
# consumer gets just the lean solver and none of the heavy dependencies.
#
# NB: the gate is a feature switch, NOT the build type. The viewer still builds under
# the Release presets, which are used for performance testing.
option(PACD_BUILD_VIEWER "Build the interactive viewer, software renderer, and dev tools" ${PROJECT_IS_TOP_LEVEL})
option(PACD_BUILD_TESTS  "Build the unit tests"                                            ${PROJECT_IS_TOP_LEVEL})
option(PACD_BUILD_BENCH  "Build the benchmarks"                                            ${PROJECT_IS_TOP_LEVEL})

# Default cache variables (can be overridden by CMakePresets.json or command line)
set(ENABLE_COMPILER_WARNINGS ON CACHE BOOL "Enable compiler warnings")
set(TREAT_WARNINGS_AS_ERRORS OFF CACHE BOOL "Treat warnings as errors")
set(ENABLE_SANITIZER_ADDRESS OFF CACHE BOOL "Enable AddressSanitizer")
set(ENABLE_SANITIZER_THREAD OFF CACHE BOOL "Enable ThreadSanitizer")
set(ENABLE_SANITIZER_UNDEFINED OFF CACHE BOOL "Enable UndefinedBehaviorSanitizer")
set(ENABLE_LTO OFF CACHE BOOL "Enable Link Time Optimization")
set(ENABLE_CLANG_TIDY OFF CACHE BOOL "Enable clang-tidy static analysis")
set(CLANG_TIDY_WARNINGS_AS_ERRORS OFF CACHE BOOL "Treat clang-tidy warnings as errors")
set(ENABLE_CPPCHECK OFF CACHE BOOL "Enable cppcheck static analysis")
set(ENABLE_COVERAGE OFF CACHE BOOL "Enable code coverage instrumentation")
set(STRICT_MODE OFF CACHE BOOL "Enable all checks and treat everything as errors")
set(ENABLE_CCACHE OFF CACHE BOOL "Enable ccache for faster rebuilds")

# ---- Primitive-approximation solver (the consumable library) ----------------
# Core convex-decomposition library: TriMesh + primitives + the decompose() API.
# No render/GL dependency and no logging dependency -- only Threads. It is C++20;
# its sole "modern" feature is std::numbers.
find_package(Threads REQUIRED)
target_add_library(pacd_solver
    src/solver/geometry.cpp
    src/solver/bvh.cpp
    src/solver/field.cpp
    src/solver/decompose.cpp
)
add_library(pacd::solver ALIAS pacd_solver)
# Export under the same spelling as the build-tree alias, so installed consumers
# (find_package) and embedded consumers (add_subdirectory/FetchContent) both link
# `pacd::solver`. Without this the export would be `pacd::pacd_solver`.
set_target_properties(pacd_solver PROPERTIES EXPORT_NAME solver)

# Public headers shipped with the library, declared as a FILE_SET so they install
# with their pacd/solver/ subpath preserved. render_bridge.hpp is deliberately
# EXCLUDED: it #includes pacd/render/mesh.hpp and only makes sense alongside the
# dev-only renderer, so it stays in-tree but is not part of the installed package.
target_sources(pacd_solver
    PUBLIC
    FILE_SET pacd_headers TYPE HEADERS
    BASE_DIRS ${PROJECT_SOURCE_DIR}/include
    FILES
        include/pacd/solver/bvh.hpp
        include/pacd/solver/field.hpp
        include/pacd/solver/geometry.hpp
        include/pacd/solver/math.hpp
        include/pacd/solver/mesh.hpp
        include/pacd/solver/primitive_sdf.hpp
        include/pacd/solver/primitives.hpp
        include/pacd/solver/solver.hpp
)
target_include_directories(pacd_solver PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_link_libraries(pacd_solver PRIVATE Threads::Threads)
target_compile_features(pacd_solver PUBLIC cxx_std_20)

# ---- Dev / viewer layer -----------------------------------------------------
# Everything below needs the GL + logging stack and is dev-only. A downstream
# consumer never builds any of it.
if(PACD_BUILD_VIEWER)
    find_package(fmt CONFIG REQUIRED)
    find_package(spdlog CONFIG REQUIRED)

    # Header-only logging library (see include/Logger.hpp). Dev-only; wired into
    # pacd_render below so spdlog/fmt are resolved explicitly instead of leaking
    # in through vcpkg's include path.
    target_add_library(logger INTERFACE)
    target_include_directories(logger INTERFACE ${PROJECT_SOURCE_DIR}/include)
    target_link_libraries(logger INTERFACE spdlog::spdlog fmt::fmt)
    target_compile_definitions(logger INTERFACE LOG_FILE="logging.txt")
    target_compile_features(logger INTERFACE cxx_std_23)

    # ---- Interactive software/OpenGL renderer ----
    # Self-contained renderer used to inspect meshes and compare a source mesh with
    # its convex decomposition. GLSL/GLFW/glad are confined to the .cpp files, so
    # only these translation units pull in the GL stack.
    find_package(glfw3 CONFIG REQUIRED)
    find_package(glad CONFIG REQUIRED)
    find_package(OpenGL REQUIRED)
    # Dear ImGui (with the GLFW + OpenGL3 backends) drives the viewer's config panel.
    # Its headers come from an IMPORTED target, so they are treated as SYSTEM and do
    # not trip -Werror / clang-tidy.
    find_package(imgui CONFIG REQUIRED)

    target_add_library(pacd_render
        src/render/math.cpp
        src/render/image.cpp
        src/render/mesh.cpp
        src/render/primitives.cpp
        src/render/stl.cpp
        src/render/camera.cpp
        src/render/scene.cpp
        src/render/scenes.cpp
        src/render/gl/shader.cpp
        src/render/gl/gpu_mesh.cpp
        src/render/viewer.cpp
    )
    target_include_directories(pacd_render PUBLIC ${PROJECT_SOURCE_DIR}/include)
    # imgui is PUBLIC: the viewer front-end includes <imgui.h> to build the panel it
    # hands to Viewer::set_on_gui. glfw/glad/GL/logger stay PRIVATE (confined to .cpp).
    target_link_libraries(pacd_render PRIVATE glfw glad::glad OpenGL::GL logger PUBLIC imgui::imgui)
    target_compile_features(pacd_render PUBLIC cxx_std_23)

    # Interactive viewer executable.
    target_add_executable(pacd-viewer src/viewer_main.cpp)
    target_link_libraries(pacd-viewer PRIVATE pacd_render pacd_solver fmt::fmt)
    target_compile_features(pacd-viewer PRIVATE cxx_std_23)

    # Measurement / regression harness and experimental prototypes (dev-only).
    add_subdirectory(tools)
    add_subdirectory(playground)
endif()

# ---- Tests & benchmarks -----------------------------------------------------
if(PACD_BUILD_TESTS)
    # Enable testing at the top level so `ctest --test-dir build/<preset>` discovers
    # the tests registered in tests/ (without this, ctest finds no root
    # CTestTestfile and silently reports success having run nothing).
    enable_testing()
    add_subdirectory(tests)
endif()

if(PACD_BUILD_BENCH)
    add_subdirectory(bench)
endif()

# ---- Install / export -------------------------------------------------------
# Only when pacd is the top-level project. A consumer embedding us via
# add_subdirectory / FetchContent uses the pacd::solver alias directly and manages
# its own install.
if(PROJECT_IS_TOP_LEVEL)
    include(CMakePackageConfigHelpers)
    set(PACD_INSTALL_CMAKEDIR ${CMAKE_INSTALL_LIBDIR}/cmake/pacd)

    install(TARGETS pacd_solver
        EXPORT pacdTargets
        FILE_SET pacd_headers DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    )
    install(EXPORT pacdTargets
        NAMESPACE pacd::
        DESTINATION ${PACD_INSTALL_CMAKEDIR}
        FILE pacdTargets.cmake
    )

    configure_package_config_file(
        ${PROJECT_SOURCE_DIR}/cmake/pacdConfig.cmake.in
        ${CMAKE_CURRENT_BINARY_DIR}/pacdConfig.cmake
        INSTALL_DESTINATION ${PACD_INSTALL_CMAKEDIR}
    )
    write_basic_package_version_file(
        ${CMAKE_CURRENT_BINARY_DIR}/pacdConfigVersion.cmake
        COMPATIBILITY SameMajorVersion
    )
    install(FILES
        ${CMAKE_CURRENT_BINARY_DIR}/pacdConfig.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/pacdConfigVersion.cmake
        DESTINATION ${PACD_INSTALL_CMAKEDIR}
    )
endif()
