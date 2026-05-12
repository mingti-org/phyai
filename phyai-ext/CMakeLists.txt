cmake_minimum_required(VERSION 3.18)
project(${SKBUILD_PROJECT_NAME} LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(
  Python
  COMPONENTS Interpreter
  REQUIRED
)
find_package(tvm_ffi CONFIG REQUIRED)

file(GLOB_RECURSE PHYAI_EXT_SOURCES CONFIGURE_DEPENDS
  "csrc/*.cpp"
  "csrc/*.cc"
  "csrc/*.cxx"
)

add_library(phyai_ext_core SHARED ${PHYAI_EXT_SOURCES})
target_include_directories(phyai_ext_core PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/include"
  "${CMAKE_CURRENT_SOURCE_DIR}/csrc"
  "${CMAKE_CURRENT_SOURCE_DIR}/third_party/xxhash"
)

tvm_ffi_configure_target(phyai_ext_core)

install(TARGETS phyai_ext_core DESTINATION ${SKBUILD_PROJECT_NAME})
tvm_ffi_install(phyai_ext_core DESTINATION ${SKBUILD_PROJECT_NAME})
