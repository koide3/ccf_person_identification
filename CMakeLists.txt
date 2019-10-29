cmake_minimum_required(VERSION 2.8.3)
project(ccf_person_identification)

add_compile_options(-std=c++11)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_LIST_DIR}/cmake")

find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Flann REQUIRED)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  roslib
  cv_bridge
  open_face_recognition
)

# set dlib directory to DLIB_ROOT
include($ENV{DLIB_ROOT}/dlib/cmake)

###################################
## catkin specific configuration ##
###################################
catkin_package(
  INCLUDE_DIRS include
  LIBRARIES ccf_person_identification
#  CATKIN_DEPENDS roscpp
#  DEPENDS system_lib
)

###########
## Build ##
###########

include_directories(
  include
  ${EIGEN3_INCLUDE_DIRS}
  ${FLANN_INCLUDE_DIRS}
  ${catkin_INCLUDE_DIRS}
)

add_library(ccf_person_identification
  src/ccf_person_identification.cpp
)
target_link_libraries(ccf_person_identification
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${FLANN_LIBRARIES}
  dlib::dlib
)
add_dependencies(ccf_person_identification ccf_person_identification_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

add_executable(ccf_person_identification_test src/ccf_person_identification_test.cpp)
add_dependencies(ccf_person_identification_test ${ccf_person_identification_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(ccf_person_identification_test
  ccf_person_identification
)
#
# add_executable(cnn_test src/cnn_test.cpp)
# add_dependencies(cnn_test ${ccf_person_identification_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
#
# target_link_libraries(cnn_test
#   ccf_person_identification
# )
