cmake_minimum_required(VERSION 3.0.2)
project(lane_tracer)

find_package(catkin REQUIRED COMPONENTS
  rospy
  sensor_msgs
  cv_bridge
  opencv4
)

catkin_package()

include_directories(
  ${catkin_INCLUDE_DIRS}
)

install(PROGRAMS
  src/lane_following_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
