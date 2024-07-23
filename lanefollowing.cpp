#include <ros/ros.h>
#include <opencv2/opencv.hpp>
#include "dynamixel_sdk.h"
#include <termios.h>
#include <fcntl.h>
#include <unistd.h>
#include <vector>
#include <algorithm>

using namespace std;
using namespace cv;

// Control table address
#define ADDR_MX_TORQUE_ENABLE           24
#define ADDR_MX_GOAL_POSITION           30
#define ADDR_MX_PRESENT_POSITION        36
#define ADDR_MX_MOVING_SPEED            32

// Data Byte Length
#define LEN_MX_GOAL_POSITION            2
#define LEN_MX_PRESENT_POSITION         2
#define LEN_MX_MOVING_SPEED             2

// Protocol version
#define PROTOCOL_VERSION                1.0

// Default setting
#define DXL1_ID                         1
#define DXL2_ID                         2
#define BAUDRATE                        2000000
#define DEVICENAME                      "/dev/ttyUSB0"

#define TORQUE_ENABLE                   1
#define TORQUE_DISABLE                  0
#define DXL_MINIMUM_POSITION_VALUE      300
#define DXL_MAXIMUM_POSITION_VALUE      600
#define DXL_MOVING_STATUS_THRESHOLD     10

#define ESC_ASCII_VALUE                 0x1b

extern "C" {
    unsigned int vel_convert(int speed);
    int syncwrite(int port_num, int group_num, int goal_velocity1, int goal_velocity2);
}

int getch()
{
    struct termios oldt, newt;
    int ch;
    tcgetattr(STDIN_FILENO, &oldt);
    newt = oldt;
    newt.c_lflag &= ~(ICANON | ECHO);
    tcsetattr(STDIN_FILENO, TCSANOW, &newt);
    ch = getchar();
    tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
    return ch;
}

int kbhit(void)
{
    struct termios oldt, newt;
    int ch;
    int oldf;

    tcgetattr(STDIN_FILENO, &oldt);
    newt = oldt;
    newt.c_lflag &= ~(ICANON | ECHO);
    tcsetattr(STDIN_FILENO, TCSANOW, &newt);
    oldf = fcntl(STDIN_FILENO, F_GETFL, 0);
    fcntl(STDIN_FILENO, F_SETFL, oldf | O_NONBLOCK);

    ch = getchar();

    tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
    fcntl(STDIN_FILENO, F_SETFL, oldf);

    if (ch != EOF)
    {
        ungetc(ch, stdin);
        return 1;
    }

    return 0;
}

string codec1 = "nvarguscamerasrc sensor-id=0 ! video/x-raw(memory:NVMM), width=(int)640, height=(int)360, format=(string)NV12, framerate=(fraction)15/1 ! \
     nvvidconv flip-method=0 ! video/x-raw, width=(int)640, height=(int)360, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink";

int main(int argc, char** argv)
{
    ros::init(argc, argv, "lanefollowing_node");
    ros::NodeHandle nh;

    // Video capture
    VideoCapture cap1(codec1, CAP_GSTREAMER);
    if (!cap1.isOpened()) {
        ROS_ERROR("Failed to open video stream");
        return -1;
    }

    // Initialize PortHandler Structs
    int port_num = portHandler(DEVICENAME);

    // Initialize PacketHandler Structs
    packetHandler();

    // Initialize Groupsyncwrite instance
    int group_num = groupSyncWrite(port_num, PROTOCOL_VERSION, ADDR_MX_MOVING_SPEED, LEN_MX_GOAL_POSITION);

    int dxl_comm_result = COMM_TX_FAIL;
    uint8_t dxl_addparam_result = false;
    int dxl_goal_position[2] = { DXL_MINIMUM_POSITION_VALUE, DXL_MAXIMUM_POSITION_VALUE };
    uint8_t dxl_error = 0;
    uint16_t dxl1_present_position = 0, dxl2_present_position = 0;

    // Open port
    if (openPort(port_num)) {
        ROS_INFO("Succeeded to open the port!");
    } else {
        ROS_ERROR("Failed to open the port!");
        return 0;
    }

    // Set port baudrate
    if (setBaudRate(port_num, BAUDRATE)) {
        ROS_INFO("Succeeded to change the baudrate!");
    } else {
        ROS_ERROR("Failed to change the baudrate!");
        return 0;
    }

    // Enable Dynamixel#1 Torque
    write1ByteTxRx(port_num, PROTOCOL_VERSION, DXL1_ID, ADDR_MX_TORQUE_ENABLE, TORQUE_ENABLE);
    if ((dxl_comm_result = getLastTxRxResult(port_num, PROTOCOL_VERSION)) != COMM_SUCCESS) {
        ROS_ERROR("%s", getTxRxResult(PROTOCOL_VERSION, dxl_comm_result));
    } else if ((dxl_error = getLastRxPacketError(port_num, PROTOCOL_VERSION)) != 0) {
        ROS_ERROR("%s", getRxPacketError(PROTOCOL_VERSION, dxl_error));
    } else {
        ROS_INFO("Dynamixel#%d has been successfully connected", DXL1_ID);
    }

    // Enable Dynamixel#2 Torque
    write1ByteTxRx(port_num, PROTOCOL_VERSION, DXL2_ID, ADDR_MX_TORQUE_ENABLE, TORQUE_ENABLE);
    if ((dxl_comm_result = getLastTxRxResult(port_num, PROTOCOL_VERSION)) != COMM_SUCCESS) {
        ROS_ERROR("%s", getTxRxResult(PROTOCOL_VERSION, dxl_comm_result));
    } else if ((dxl_error = getLastRxPacketError(port_num, PROTOCOL_VERSION)) != 0) {
        ROS_ERROR("%s", getRxPacketError(PROTOCOL_VERSION, dxl_error));
    } else {
        ROS_INFO("Dynamixel#%d has been successfully connected", DXL2_ID);
    }

    Mat frame1, gray, dst;
    Point2d prevpt1(110, 60);
    Point2d prevpt2(520, 60);
    Point2d cpt[2];
    Point2d fpt;
    int minlb[2];
    int thres;
    double ptdistance[2];
    double threshdistance[2];
    vector<double> mindistance1;
    vector<double> mindistance2;
    double error;
    double myproms;
    int count = 0;

    while (ros::ok())
    {
        int64 t1 = getTickCount();

        cap1 >> frame1;

        cvtColor(frame1, gray, COLOR_BGR2GRAY);
        gray = gray + 100 - mean(gray)[0];
        thres = 160;
        threshold(gray, gray, thres, 255, THRESH_BINARY);

        dst = gray(Rect(0, gray.rows / 3 * 2, gray.cols, gray.rows / 3));

        Mat labels, stats, centroids;
        int cnt = connectedComponentsWithStats(dst, labels, stats, centroids);
        if (cnt > 1) {
            for (int i = 1; i < cnt; i++) {
                double* p = centroids.ptr<double>(i);
                ptdistance[0] = abs(p[0] - prevpt1.x);
                ptdistance[1] = abs(p[0] - prevpt2.x);
                mindistance1.push_back(ptdistance[0]);
                mindistance2.push_back(ptdistance[1]);
           
