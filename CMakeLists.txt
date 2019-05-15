cmake_minimum_required(VERSION 3.1)
project(example)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# libigl
option(LIBIGL_WITH_OPENGL            "Use OpenGL"         ON)
option(LIBIGL_WITH_OPENGL_GLFW       "Use GLFW"           ON)
option(LIBIGL_WITH_VIEWER            "Use OpenGL viewer"  ON)
option(LIBIGL_WITH_OPENGL_GLFW_IMGUI "Use Imgui"          ON)

find_package(LIBIGL REQUIRED QUIET)
find_package(Portaudio REQUIRED)

# Add your project files
file(GLOB SRCFILES *.cpp)
file(GLOB TOOLS tools/*.cpp)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

foreach(TOOL_NAME ${TOOLS})
    get_filename_component(TOOL_BIN ${TOOL_NAME} NAME_WE)
    set(TOOL_BIN ${TOOL_BIN}_bin)
    message("-- Building tool : ${TOOL_BIN}")
    add_executable(${TOOL_BIN} ${SRCFILES} ${TOOL_NAME})
    target_link_libraries(${TOOL_BIN}
        igl::core
        igl::opengl_glfw
        igl::opengl_glfw_imgui
        ${PORTAUDIO_LIBRARIES})
endforeach()
