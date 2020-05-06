cmake_minimum_required(VERSION 3.14.0 FATAL_ERROR)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules")
project(lightvis-utils VERSION 0.1.0 LANGUAGES CXX)

add_executable(utils-main
    ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
)

target_link_libraries(utils-main
    PRIVATE
        depends::nuklear
        lightvis
)
