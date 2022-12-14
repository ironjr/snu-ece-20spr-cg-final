cmake_minimum_required(VERSION 3.10)
cmake_policy(VERSION 3.10)
project("Final" C CXX)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/../bin)

if(WIN32)
  include_directories(glad/include)
  include_directories("C:/OpenGL/includes")
  link_directories("C:/OpenGL/lib")
  set(SOURCE_FILES main.cpp glad/src/glad.c)
  add_executable(main ${SOURCE_FILES})
  target_link_libraries(main opengl32 glfw3 assimp FreeImage)
endif(WIN32)

if(UNIX)
  find_package(PkgConfig REQUIRED)
  pkg_search_module(GLFW REQUIRED glfw3)
  find_package(glm REQUIRED)
  find_package(assimp REQUIRED)
  # find_package(freeimage REQUIRED)

  # I manually included FreeImage library, since my apt does not work properly.
  find_path(
    FREEIMAGE_INCLUDE_DIRS FreeImage.h
    HINTS /usr/include # ${FreeImage_ROOT}
  )
  find_library(
    FREEIMAGE_LIBRARIES
    NAMES freeimage
    HINTS /usr/lib # ${FreeImage_ROOT}
  )
  include(FindPackageHandleStandardArgs)
  find_package_handle_standard_args(
    freeimage DEFAULT_MSG FREEIMAGE_INCLUDE_DIRS FREEIMAGE_LIBRARIES
  )

  include_directories(${GLFW_INCLUDE_DIRS})
  include_directories(${GLM_INCLUDE_DIRS})
  include_directories(${ASSIMP_INCLUDE_DIRS})
  include_directories(${FREEIMAGE_INCLUDE_DIRS})
  include_directories(${CMAKE_SOURCE_DIR}/glad/include)
  set(SOURCE_FILES main.cpp glad/src/glad.c)
  add_executable(main ${SOURCE_FILES})
  target_link_libraries(
    main
    ${GLFW_STATIC_LIBRARIES}
    ${ASSIMP_LIBRARIES}
    ${FREEIMAGE_LIBRARIES}
  )
endif(UNIX)
