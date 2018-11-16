cmake_minimum_required(VERSION 3.1)
project(example)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# libigl
option(LIBIGL_WITH_OPENGL            "Use OpenGL"         ON)
option(LIBIGL_WITH_OPENGL_GLFW       "Use GLFW"           ON)
option(LIBIGL_WITH_VIEWER            "Use OpenGL viewer"  ON)

find_package(LIBIGL REQUIRED QUIET)
find_package(Portaudio REQUIRED)

# Add your project files
file(GLOB SRCFILES *.cpp)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
add_executable(simulator tools/simulator.cpp ${SRCFILES})
target_link_libraries(simulator igl::core igl::opengl_glfw
    ${PORTAUDIO_LIBRARIES})
