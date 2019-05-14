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
file(GLOB TOOLS tools/*.cpp)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

foreach(TOOL_NAME ${TOOLS})
    get_filename_component(TOOL_BIN ${TOOL_NAME} NAME_WE)
    set(TOOL_BIN ${TOOL_BIN}_bin)
    message("-- Building tool : ${TOOL_BIN}")
    add_executable(${TOOL_BIN} ${SRCFILES} ${TOOL_NAME})
    target_link_libraries(${TOOL_BIN} igl::core igl::opengl_glfw
        ${PORTAUDIO_LIBRARIES})
endforeach()


# add_executable(simulator tools/simulator.cpp ${SRCFILES})
# add_executable(register_segmentation tools/register_segmentation.cpp ${SRCFILES})
# add_executable(viz_segmentation tools/viz_segmentation.cpp ${SRCFILES})
# target_link_libraries(simulator igl::core igl::opengl_glfw
#     ${PORTAUDIO_LIBRARIES})
# target_link_libraries(register_segmentation igl::core)
# target_link_libraries(viz_segmentation igl::core igl::opengl_glfw)
