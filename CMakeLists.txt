cmake_minimum_required(VERSION 3.21)
set(CMAKE_CXX_STANDARD 20)

project(micrograd VERSION 0.1)

include(CTest)
Include(FetchContent)

FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG        v3.5.4 # or a different version
)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

FetchContent_MakeAvailable(Catch2)

include_directories (${CMAKE_SOURCE_DIR}/include)

add_library             ( engine lib/engine.cpp)
target_link_libraries   ( engine )

add_executable( tests tests/tests.cpp )
target_link_libraries( tests PRIVATE Catch2::Catch2WithMain engine )
add_test( NAME engine COMMAND tests )

add_executable          ( micrograd src/micrograd.cpp )
target_link_libraries   ( micrograd engine )