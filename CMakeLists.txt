cmake_minimum_required(VERSION 3.15)
project(lbmcl)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenGL 3.3 REQUIRED)
find_package(OpenCL 1.2 REQUIRED)

add_library(glad STATIC "thirdparty/src/glad.c")
target_include_directories(glad PRIVATE "thirdparty/include")

file(GLOB CXX_SOURCES "src/*.cpp")

add_executable(lbmcl ${CXX_SOURCES})
target_include_directories(lbmcl PRIVATE "thirdparty/include" ${OpenCL_INCLUDE_DIRS})
target_link_libraries(lbmcl PRIVATE ${OPENGL_gl_LIBRARY} ${OpenCL_LIBRARIES} glad)
target_link_libraries(lbmcl PRIVATE "${CMAKE_SOURCE_DIR}/thirdparty/lib/glfw3.lib")
