#include <iostream>
#include <algorithm>
#include <fstream>
#include <chrono>
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <nav_msgs/Odometry.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/core/core.hpp>
#include "ground_tracker.h"
#include "simple_compensation.hpp"

using namespace std;

queue<pair<double, cv::Mat>> img_queue;
queue<pair<double, Eigen::Matrix4d>> pose_queue;

ros::Subscriber sub_img;
ros::Subscriber sub_pose;
image_transport::Publisher pub_img_track;
image_transport::Publisher pub_ipm_track;
shared_ptr<gv::GroundTracker> tracker;

std::map<double, double> all_imu_pitch;
bool enable_pitch_comp;
double pitch_comp_windowsize;
Eigen::Matrix4d last_pose;

void img_callback(const sensor_msgs::ImageConstPtr &msg)
{
    cv::Mat mm = cv_bridge::toCvCopy(msg, "bgr8")->image;
    img_queue.push(make_pair(msg->header.stamp.toSec(), mm));

    // Note: For VIO implementation, process IMU data here.
    // Feed the pose prediction into feature tracker.

    ros::Rate wait_for_pose(100);
    while (pose_queue.empty() && ros::ok())
    {
        ROS_WARN("Waiting for pose!!!");
        wait_for_pose.sleep();
    }

    // **Attention**: Assuming synced image and pose inputs here.
    while (!pose_queue.empty() && !img_queue.empty()&& ros::ok())
    {
        if (pose_queue.front().first <= img_queue.front().first - 1e-2)
            pose_queue.pop();
        else if (pose_queue.front().first >= img_queue.front().first + 1e-2)
            img_queue.pop();
        else
        {   
            ROS_INFO("Get synced data!!! %.5lf %.5lf",pose_queue.front().first, img_queue.front().first);

            double tt = pose_queue.front().first;
            cv::Mat img = img_queue.front().second;
            Eigen::Matrix4d Twik = pose_queue.front().second;
            Eigen::Matrix4d Twik_1 = last_pose;
            Eigen::Matrix4d Tckck_1 = (Twik * tracker->config.Tic).inverse() * (Twik_1 * tracker->config.Tic);

            pose_queue.pop();
            img_queue.pop();

            gv::CameraGroundGeometry cg = tracker->config.cg;

            // Calculate temporary camera-ground geometry considering
            // high-frequency pitch compenstation
            if (enable_pitch_comp)
            {
                all_imu_pitch[tt] = gv::m2att(Twik.block<3, 3>(0, 0)).x() * 180 / M_PI;
                double comp = simple_pitch_compenstation(tt, all_imu_pitch, pitch_comp_windowsize);
                cg.update(cg.getAlpha(), cg.getTheta() + comp);
            }
            // main feature tracking
            cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
            auto feature_frame = tracker->trackImage(tt, img, 10.0, cg, Tckck_1, false);

            // visualization
            std_msgs::Header header;
            header.stamp = ros::Time(tt);
            header.frame_id = "cam0";
            sensor_msgs::ImagePtr img_show_msg = cv_bridge::CvImage(header, "bgr8", tracker->cur_img_show).toImageMsg();
            sensor_msgs::ImagePtr ipm_show_msg = cv_bridge::CvImage(header, "bgr8", tracker->cur_ipm_show).toImageMsg();
            pub_img_track.publish(img_show_msg);
            pub_ipm_track.publish(ipm_show_msg);

            last_pose = Twik;
        }
    }
}

void imu_pose_callback(const nav_msgs::Odometry &msg)
{
    Eigen::Matrix4d pose = Eigen::Matrix4d::Identity();
    pose.block<3, 3>(0, 0) = Eigen::Quaterniond(
                                 msg.pose.pose.orientation.w,
                                 msg.pose.pose.orientation.x,
                                 msg.pose.pose.orientation.y,
                                 msg.pose.pose.orientation.z)
                                 .toRotationMatrix();
    pose.block<3, 1>(0, 3) = Eigen::Vector3d(msg.pose.pose.position.x,
                                             msg.pose.pose.position.y,
                                             msg.pose.pose.position.z);
    pose_queue.push(make_pair(msg.header.stamp.toSec(), pose));
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "ground_tracker_node");
    ros::NodeHandle nh("~");
    
    string config_path;
    nh.getParam("config_path", config_path);
    ROS_WARN("config_path: %s\n",config_path.c_str());
    cv::FileStorage fsSettings(config_path, cv::FileStorage::READ);
    enable_pitch_comp = (int)fsSettings["enable_pitch_comp"];
    pitch_comp_windowsize = fsSettings["pitch_comp_windowsize"];
    fsSettings.release();

    gv::CameraConfig config(config_path);
    tracker = make_shared<gv::GroundTracker>(config);

    ros::AsyncSpinner spinner(0);
    spinner.start();

    sub_img = nh.subscribe("/cam0/image_raw", 1, img_callback);
    sub_pose = nh.subscribe("/pose_gt", 1, imu_pose_callback);
    image_transport::ImageTransport it(nh);
    pub_img_track = it.advertise("img_track", 2);
    pub_ipm_track = it.advertise("ipm_track", 2);

    ros::waitForShutdown();
}

