#ifndef GROUND_TRACKER_H
#define GROUND_TRACKER_H

#include <cstdio>
#include <iostream>
#include <queue>
#include <csignal>
#include "gv_utils.h"
#include "ipm_processer.h"
#include <opencv2/features2d.hpp>

namespace gv
{

	using namespace std;
	using namespace Eigen;

	enum GroundFeatureMode
	{
		MINDIST = 0, // 1) Min dist-based feature extraction, referring to VINS-Mono.
		GRID = 1     // 2) Grid-based feature extraction, inspried by https://github.com/KumarRobotics/msckf_vio.
	};

	/**
	 * @brief Ground tracker for ground feature extraction and tracking.
	 *
	 * The ground tracker could work in either IPM or perspective mode.
	 * In both modes, the camera-ground geometry is used for feature prediction.
	 *
	 */
	class GroundTracker
	{
	public:
		GroundTracker(CameraConfig conf);
		~GroundTracker(){};

		/**
		 * Main function for ground feature tracking via IPM.
		 *
		 * @param _cur_time Current timestamp.
		 * @param _img Input image.
		 * @param threshold Outlier threshold to check optical flow with prediction.
		 * @param cg Current camera-ground geometry.
		 * @param Tckck_1 Relative pose of the camera.
		 * @param show_track Enable cv::imshow().
		 * @return Features (normalized coords in the perspective image) in the current frame.
		 *
		 */
		map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> trackImage(double _cur_time,
																			const cv::Mat &_img, const double &threshold, const CameraGroundGeometry &cg,
																			const Eigen::Matrix4d &Tckck_1, const bool &show_track = true);

		/**
		 * Main function for ground feature tracking on the perspective image (without IPM).
		 *
		 * @param _cur_time Current timestamp.
		 * @param _img Input image.
		 * @param threshold Outlier threshold to check optical flow with prediction.
		 * @param cg Current camera-ground geometry.
		 * @param Tckck_1 Relative pose of the camera.
		 * @param show_track Enable cv::imshow().
		 * @return Features (normalized coords in the perspective image) in the current frame.
		 *
		 */
		map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> trackImagePerspective(double _cur_time,
																					   const cv::Mat &_img, const double &threshold, const CameraGroundGeometry &cg,
																					   const Eigen::Matrix4d &Tckck_1, const bool &show_track = true);

		bool inBorder(const cv::Point2f &pt);
		void reduceVector(vector<cv::Point2f> &v, vector<uchar> status);
		void reduceVector(vector<int> &v, vector<uchar> status);
		double distance(cv::Point2f &pt1, cv::Point2f &pt2);

		int row, col;
		cv::Mat mask;
		cv::Mat cur_img_semantic;
		cv::Mat prev_img, cur_img, cur_img_show;
		cv::Mat prev_ipm, cur_ipm, cur_ipm_show;
		vector<cv::Point2f> n_pts;
		vector<cv::Point2f> predict_pts;
		vector<cv::Point2f> prev_pts, cur_pts;
		vector<int> ids;
		vector<int> track_cnt;
		double prev_time;
		int n_id = 1000000000; // to identify the ground features

		CameraConfig config;
		shared_ptr<IPMProcesser> ipmproc;
		cv::Mat ground_mask;

		CameraGroundGeometry prev_cg;
		CameraGroundGeometry cur_cg;

		GroundFeatureMode feature_mode = GroundFeatureMode::MINDIST;

		// Min dist-based feature extraction.
		int MAX_CNT = 30;
		int MIN_DIST = 20;

		// Grid-based feature extraction.
		cv::Ptr<cv::Feature2D> detector_ptr;
		int GRID_ROW = 8;
		int GRID_COL = 4;
		int GRID_MIN_FEATURE_NUM = 2;
		int GRID_MAX_FEATURE_NUM = 3;
		int GRID_HEIGHT;
		int GRID_WIDTH;
		map<int, vector<tuple<int, cv::Point2f>>> feature2grid(
			vector<cv::Point2f> pts, vector<int> ids);

	};
}

#endif