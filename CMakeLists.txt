cmake_minimum_required(VERSION 3.1)
set(CMAKE_CXX_STANDARD 17)

project(DummyProject VERSION 0.1)

include(CTest)
Include(FetchContent)

FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG        v3.3.2 # or a different version
)

FetchContent_MakeAvailable(Catch2)

include_directories (${CMAKE_SOURCE_DIR}/include)

add_library             ( DummyLib lib/DummyLib.cpp)
target_link_libraries   ( DummyLib )

add_executable(DummyTest tests/DummyTest.cpp)
target_link_libraries(DummyTest PRIVATE Catch2::Catch2WithMain DummyLib )
add_test(NAME DummyTest COMMAND DummyTest)

add_executable          ( DummyExec src/Dummy.cpp )
target_link_libraries   ( DummyExec DummyLib )