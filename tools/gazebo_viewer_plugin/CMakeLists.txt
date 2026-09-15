cmake_minimum_required(VERSION 3.17)
project(ErcViewerLighting LANGUAGES CXX)

find_package(gz-gui8 REQUIRED)
find_package(gz-rendering8 REQUIRED)
find_package(gz-plugin2 REQUIRED COMPONENTS register)
find_package(Qt5 REQUIRED COMPONENTS Core Quick)

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
add_library(ErcViewerLighting SHARED
  ErcViewerLighting.cc ErcViewerLighting.hh ErcViewerLighting.qrc)
target_compile_features(ErcViewerLighting PRIVATE cxx_std_17)
target_compile_options(ErcViewerLighting PRIVATE -Wall -Wextra -Wpedantic)
target_link_libraries(ErcViewerLighting PRIVATE
  gz-gui8::gz-gui8 gz-rendering8::gz-rendering8 gz-plugin2::register
  Qt5::Core Qt5::Quick)
