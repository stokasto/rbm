cmake_minimum_required(VERSION 2.8)

PROJECT(RBM)

SET( CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_MODULE_PATH})

FIND_PACKAGE(Eigen3 REQUIRED)
FIND_PACKAGE(GTest)

INCLUDE_DIRECTORIES(${EIGEN3_INCLUDE_DIR})
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/include)

#set the default path for built executables to the "bin" directory
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
#set the default path for built libraries to the "lib" directory
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

# ----- Testing -----
add_executable(test_8_5_8 test/test_8_5.cpp)
IF (GTEST_FOUND)
  ENABLE_TESTING()
  include_directories(${GTEST_INCLUDE_DIR})
ENDIF (GTEST_FOUND)
