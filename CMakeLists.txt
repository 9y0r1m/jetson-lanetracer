cmake_minimum_required(VERSION 3.0.2)
project(lanefollowing)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  cv_bridge
  image_transport
)

find_package(OpenCV 4 REQUIRED)
find_package(dynamixel_sdk REQUIRED)

catkin_package(
  CATKIN_DEPENDS roscpp std_msgs cv_bridge image_transport
  DEPENDS OpenCV dynamixel_sdk
)

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${dynamixel_sdk_INCLUDE_DIRS}
)

add_executable(${PROJECT_NAME}_node src/lanefollowing.cpp)

target_link_libraries(${PROJECT_NAME}_node
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${dynamixel_sdk_LIBRARIES}
)
