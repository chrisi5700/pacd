# Standalone downstream project that consumes an installed pacd via find_package.
# Build it against an install tree:
#   cmake --install build/core --prefix /path/to/prefix
#   cmake -S examples/consumer -B build/consumer -DCMAKE_PREFIX_PATH=/path/to/prefix
#   cmake --build build/consumer && ./build/consumer/pacd_consumer
cmake_minimum_required(VERSION 3.28)
project(pacd_consumer LANGUAGES CXX)

find_package(pacd CONFIG REQUIRED)

add_executable(pacd_consumer main.cpp)
target_link_libraries(pacd_consumer PRIVATE pacd::solver)
target_compile_features(pacd_consumer PRIVATE cxx_std_20)
