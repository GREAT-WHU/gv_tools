cmake_minimum_required(VERSION 2.8.12)

# Project name
project(gv_tools)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -MP")
add_compile_options(-MP)

add_definitions(-DGLOG_NO_ABBREVIATED_SEVERITIES)
add_definitions(-DNOMINMAX)

if(NOT WIN32)
  string(ASCII 27 Esc)
  set(ColourReset "${Esc}[m")
  set(Red         "${Esc}[31m")
endif()

if(DEFINED CATKIN_DEVEL_PREFIX)
   message("${Red}!!!!!!!ROS workspace detected, building ROS node...!!!!!!!${ColourReset}")
else()
   message("${Red}!!!!!!!Building non-ROS libraries and examples...!!!!!!! ${ColourReset}")
endif()

find_package(OpenCV)
   if(NOT OpenCV_FOUND)
      message(FATAL_ERROR "OpenCV not found.")
   endif()
find_package(Eigen3 REQUIRED)
message(STATUS "OPENCV: " ${OpenCV_VERSION} )

include_directories(
    ./
	./include
    ${EIGEN3_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
)
file(GLOB_RECURSE SRC_FILE_LIST "src/*.cpp" "src/*.cc" "src/*.c" "camodocal/*.cpp" "camodocal/*.cc" "camodocal/*.c")
file(GLOB_RECURSE HEARDER_FILE_LIST "include/*.h"  "include/*.hpp")

list(APPEND LIBS
    ${OpenCV_LIBS}
)
add_library(gv_tools SHARED ${SRC_FILE_LIST})
add_executable(track_dataset example/track_dataset.cpp)
target_link_libraries(track_dataset gv_tools ${LIBS})

if(DEFINED CATKIN_DEVEL_PREFIX)
    find_package (catkin REQUIRED COMPONENTS
       roscpp
       rospy
       std_msgs
       cv_bridge
       image_transport
       tf
       sensor_msgs
    )
    catkin_package (
       CATKIN_DEPENDS roscpp rospy std_msgs cv_bridge image_transport tf sensor_msgs 
       LIBRARIES gv_tools
    )
    include_directories(
       ${catkin_INCLUDE_DIRS}
    )
    add_executable (ground_tracker_node
       ros/ground_tracker_node.cpp
    )
    target_link_libraries(ground_tracker_node
       ${LIBS} ${catkin_LIBRARIES} gv_tools
    )
endif()




