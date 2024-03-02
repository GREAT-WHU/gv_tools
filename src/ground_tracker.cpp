#include "ground_tracker.h"

#define KLT_PATCH_SIZE 21

namespace gv
{
	GroundTracker::GroundTracker(CameraConfig conf) : config(conf)
	{
		ipmproc = shared_ptr<IPMProcesser>(new IPMProcesser(conf, IPMType::NORMAL));
		switch(conf.feature_mode)
		{
			case 0:feature_mode= GroundFeatureMode::MINDIST; break;
			case 1:feature_mode= GroundFeatureMode::GRID; break;
			default: feature_mode= GroundFeatureMode::MINDIST; break;
		}
		ground_mask = cv::Mat();
		MIN_DIST = conf.min_dist_ground;
		MAX_CNT = conf.max_cnt_ground;

		cur_cg = conf.cg;
		prev_cg = conf.cg; // prior camera-ground geometry

		if (feature_mode == GroundFeatureMode::GRID)
		{
			GRID_ROW = config.grid_row;
			GRID_COL = config.grid_col;
			GRID_MIN_FEATURE_NUM = config.grid_min_feature_num;
			GRID_MAX_FEATURE_NUM = config.grid_max_feature_num;
			GRID_HEIGHT = config.IPM_HEIGHT / GRID_ROW;
			GRID_WIDTH = config.IPM_WIDTH / GRID_COL;
			detector_ptr = cv::FastFeatureDetector::create(5.0);

		}
	};

	map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> GroundTracker::trackImage(double _cur_time,
																					   const cv::Mat &_img, const double &threshold, const CameraGroundGeometry &cg,
																					   const Eigen::Matrix4d &Tckck_1, const bool &show_track)
	{
		cur_pts.clear();
		cur_cg = cg;

		cv::Mat img_undist = ipmproc->genUndistortedImage(_img);
		cv::Vec4d intrinsics = ipmproc->getUndistortedIntrinsics();

		cur_img = img_undist;

		cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
		clahe->apply(cur_img, cur_img);

		// Add new features only in the top half region for good tracking.
		mask = cv::Mat(config.IPM_HEIGHT, config.IPM_WIDTH, CV_8UC1);
		mask.setTo(0);
		mask(cv::Rect(0, 0, config.IPM_WIDTH, config.IPM_HEIGHT / 2)).setTo(255);

		if (show_track)
			cv::imshow("mask", mask);

		if (!(cur_cg == prev_cg))
		{
			ipmproc->updateCameraGroundGeometry(cur_cg);
		}
		cur_ipm = ipmproc->genIPM(cur_img, false);

		if (!cur_img_semantic.empty())
		{
			vector<cv::Mat> img_channels_temp;
			cv::split(cur_img_semantic, img_channels_temp);
			cur_img_semantic = img_channels_temp[2];
			cur_img_semantic.setTo(255, cur_img_semantic == 7);
			cur_img_semantic.setTo(255, cur_img_semantic == 6);
			cur_img_semantic.setTo(255, cur_img_semantic == 20);
			cv::Mat cur_ipm_semantic = ipmproc->genIPM(cur_img_semantic, true);
			mask.setTo(0, cur_ipm_semantic != 255);
		}

		// For visualization
		cv::cvtColor(cur_ipm, cur_ipm_show, cv::COLOR_GRAY2BGR);
		cv::cvtColor(img_undist, cur_img_show, cv::COLOR_GRAY2BGR);

		vector<cv::Point2f> pred_pts;
		vector<cv::Point2f> prev_ipm_pts, cur_ipm_pts;
		vector<uchar> status;
		vector<float> err;

		if (prev_pts.size() > 0)
		{
			Eigen::Matrix3d Rckck_1 = Tckck_1.block<3, 3>(0, 0);
			Eigen::Vector3d tckck_1 = Tckck_1.block<3, 1>(0, 3);

			// Predict tracked points
			for (int i = 0; i < prev_pts.size(); i++)
			{
				cv::Point2f p_ipm = ipmproc->Perspective2IPM(prev_pts[i], &prev_cg);
				prev_ipm_pts.push_back(p_ipm);
				Eigen::Vector3d pc0k_1f = ipmproc->Perspective2Metric(prev_pts[i], &prev_cg);
				Eigen::Vector3d pc0kf = Rckck_1 * pc0k_1f + tckck_1;
				cv::Point2f uvc1kf = ipmproc->Metric2IPM(pc0kf);
				pred_pts.push_back(uvc1kf);
			}

			cur_ipm_pts = pred_pts;

			// Optical flow tracking

			if (threshold > 10)
				cv::calcOpticalFlowPyrLK(prev_ipm, cur_ipm, prev_ipm_pts, cur_ipm_pts, status, err, cv::Size(KLT_PATCH_SIZE, KLT_PATCH_SIZE), 2,
										 cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01), cv::OPTFLOW_USE_INITIAL_FLOW);
			else
				cv::calcOpticalFlowPyrLK(prev_ipm, cur_ipm, prev_ipm_pts, cur_ipm_pts, status, err, cv::Size(KLT_PATCH_SIZE, KLT_PATCH_SIZE), 1,
										 cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01), cv::OPTFLOW_USE_INITIAL_FLOW);

			if (true) // flow back
			{
				vector<uchar> reverse_status;
				vector<cv::Point2f> reverse_pts = prev_ipm_pts;
				cv::calcOpticalFlowPyrLK(cur_ipm, prev_ipm, cur_ipm_pts, reverse_pts, reverse_status, err, cv::Size(KLT_PATCH_SIZE, KLT_PATCH_SIZE), 1,
										 cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01), cv::OPTFLOW_USE_INITIAL_FLOW);
				for (size_t i = 0; i < status.size(); i++)
				{
					if (status[i] && reverse_status[i] && distance(prev_ipm_pts[i], reverse_pts[i]) <= 0.5 && distance(cur_ipm_pts[i], pred_pts[i]) <= threshold)
					{
						status[i] = 1;
					}
					else
						status[i] = 0;
				}
			}

			for (int i = 0; i < cur_ipm_pts.size(); i++)
			{
				cur_pts.push_back(ipmproc->IPM2Perspective(cur_ipm_pts[i]));
			}

			reduceVector(prev_pts, status);
			reduceVector(cur_pts, status);
			reduceVector(prev_ipm_pts, status);
			reduceVector(cur_ipm_pts, status);
			reduceVector(ids, status);
			reduceVector(track_cnt, status);

			// Coarse speed estimation
			if (prev_ipm_pts.size() > 0)
			{
				vector<double> d_pt;
				for (int i = 0; i < prev_ipm_pts.size(); i++)
					d_pt.push_back(cv::norm(cur_ipm_pts[i] - prev_ipm_pts[i]) * config.IPM_RESO);
				std::sort(d_pt.begin(), d_pt.end());
				std::cerr << "Coarse speed estimation: " << d_pt[d_pt.size() / 2] / (_cur_time - prev_time) << " (m/s)" << std::endl;
			}

			for (int i = 0; i < pred_pts.size(); i++)
			{
				if (status[i] == 1)
					cv::circle(cur_ipm_show, pred_pts[i], 5, cv::Scalar(0, 255, 0), 1);
				else
					cv::circle(cur_ipm_show, pred_pts[i], 5, cv::Scalar(255, 0, 0), 1);
			}

			for (int i = 0; i < cur_pts.size(); i++)
			{
				cv::circle(cur_ipm_show, cur_ipm_pts[i], 0, cv::Scalar(0, 255, 0), 4);
				cv::line(cur_ipm_show, prev_ipm_pts[i], cur_ipm_pts[i], cv::Scalar(0, 255, 0));
				if (feature_mode == GroundFeatureMode::MINDIST)
				{
					cv::circle(mask, cur_ipm_pts[i], MIN_DIST, 0, -1);
				}
				cv::circle(cur_img_show, cur_pts[i], 0, cv::Scalar(0, 255, 0), 4);
				cv::line(cur_img_show, prev_pts[i], cur_pts[i], cv::Scalar(0, 255, 0), 1);
			}

			if (cur_pts.size() >= 8)
			{
				printf("HM ransac begins\n");

				vector<uchar> mask_outlier;
				cv::Mat Hmatrix = cv::findHomography(cur_ipm_pts, prev_ipm_pts, mask_outlier, cv::RANSAC, 5.0);
				// cv::Mat Hmatrix = cv::findHomography(cur_ipm_pts, prev_ipm_pts, mask_outlier, cv::LMEDS);
				int size_a = cur_pts.size();
				printf("HM ransac: %d -> %lu: %f\n", size_a, cur_pts.size(), 1.0 * cur_pts.size() / size_a);

				for (int jj = 0; jj < mask_outlier.size(); jj++)
				{
					if ((int)mask_outlier[jj] == 0)
						cv::circle(cur_ipm_show, cur_ipm_pts[jj], 10, cv::Scalar(0, 0, 255), 1);
				}

				reduceVector(prev_pts, mask_outlier);
				reduceVector(cur_pts, mask_outlier);
				reduceVector(prev_ipm_pts, mask_outlier);
				reduceVector(cur_ipm_pts, mask_outlier);
				reduceVector(ids, mask_outlier);
				reduceVector(track_cnt, mask_outlier);
			}
		}

		vector<cv::Point2f> n_pts_ipm;
		// 1) Min dist-based feature extraction, referring to VINS-Mono.
		if (feature_mode == GroundFeatureMode::MINDIST)
		{
			// Extract new points
			int n_max_cnt = MAX_CNT - static_cast<int>(cur_pts.size());
			if (n_max_cnt > 0)
			{
				if (mask.empty())
					cout << "mask is empty " << endl;
				if (mask.type() != CV_8UC1)
					cout << "mask type wrong " << endl;
				cv::goodFeaturesToTrack(cur_ipm, n_pts_ipm, MAX_CNT - cur_pts.size(), 0.001, MIN_DIST, mask, 3, true);
			}
		}
		// 2) Grid-based feature extraction, inspried by https://github.com/KumarRobotics/msckf_vio.
		else if (feature_mode == GroundFeatureMode::GRID)
		{
			auto cur_grid_features = feature2grid(cur_ipm_pts, ids);

			for (const auto &feature : cur_ipm_pts)
			{
				const cv::Point2f &pt = feature;
				const int y = static_cast<int>(pt.y);
				const int x = static_cast<int>(pt.x);
				if(x<0 || y<0 || x>=cur_ipm.cols || y>=cur_ipm.rows) continue;

				int up_lim = y - 2, bottom_lim = y + 3, left_lim = x - 2, right_lim = x + 3;
				if (up_lim < 0)
					up_lim = 0;
				if (bottom_lim > cur_ipm.rows) 
					bottom_lim = cur_ipm.rows;
				if (left_lim < 0)              
					left_lim = 0;
				if (right_lim > cur_ipm.cols)  
					right_lim = cur_ipm.cols;
				mask(cv::Range(up_lim, bottom_lim), cv::Range(left_lim, right_lim)) = 0;
			}

			// Detect new features.
			vector<cv::KeyPoint> new_features(0);
			detector_ptr->detect(cur_ipm, new_features, mask);

			// Group the features into grids
			map<int, vector<cv::KeyPoint>> grid_new_features;
			for (int code = 0; code < GRID_ROW * GRID_COL; ++code)
				grid_new_features[code] = vector<cv::KeyPoint>(0);

			for (int i = 0; i < new_features.size(); ++i)
			{
				const cv::Point2f &pt = new_features[i].pt;
				int row = static_cast<int>(pt.y / GRID_HEIGHT);
				int col = static_cast<int>(pt.x / GRID_WIDTH);
				int code = row * GRID_COL + col;
				grid_new_features[code].push_back(new_features[i]);
			}

			// Sort the new features in each grid based on its response.
			for (auto &key_item : grid_new_features)
			{	
				auto & item = key_item.second;
				if (item.size() > GRID_MAX_FEATURE_NUM)
				{
					std::sort(item.begin(), item.end(),
					[](cv::KeyPoint pt1, cv::KeyPoint pt2){return pt1.response > pt2.response;});
					item.erase(
						item.begin() + GRID_MAX_FEATURE_NUM, item.end());
				}
			}

			int new_added_feature_num = 0;
			// Collect new features within each grid with high response.
			for (int code = 0; code < GRID_ROW * GRID_COL; ++code)
			{
				const auto &features_this_grid = cur_grid_features[code];
				const auto &new_features_this_grid = grid_new_features[code];

				if (features_this_grid.size() >= GRID_MIN_FEATURE_NUM)
					continue;

				int vacancy_num = GRID_MIN_FEATURE_NUM -
								  features_this_grid.size();
				for (int k = 0;
					 k < vacancy_num && k < new_features_this_grid.size(); ++k)
				{
					n_pts_ipm.push_back(new_features_this_grid[k].pt);
					++new_added_feature_num;
				}
			}
		}

		for (auto &p_ipm : n_pts_ipm)
		{
			cv::Point2f p = ipmproc->IPM2Perspective(p_ipm);
			n_pts.push_back(p);
			cur_pts.push_back(p);
			cur_ipm_pts.push_back(p_ipm);
			ids.push_back(n_id++);
			track_cnt.push_back(1);

			cv::circle(cur_ipm_show, p_ipm, 0, cv::Scalar(0, 0, 255), 4);
			cv::circle(cur_img_show, p, 0, cv::Scalar(0, 0, 255), 4);
		}

		if (show_track)
		{
			cv::imshow("cur_ipm_show", cur_ipm_show);
			cv::imshow("cur_img_show", cur_img_show);
			cv::waitKey(1);
		}
		// cv::imwrite("/home/zhouyuxuan/data/cur_ipm_show_"+to_string((long long)(_cur_time*1e9))+".png",cur_ipm_show);
		// cv::imwrite("/home/zhouyuxuan/data/cur_img_show_"+to_string((long long)(_cur_time*1e9))+".png",cur_img_show);


		map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> featureFrame;
		for (size_t i = 0; i < ids.size(); i++)
		{
			int feature_id = ids[i];
			cv::Point2f uvc0 = cur_pts[i];
			double x, y, z;
			x = (uvc0.x - intrinsics(2)) / intrinsics(0);
			y = (uvc0.y - intrinsics(3)) / intrinsics(1);
			z = 1;
			int camera_id = 0;

			Eigen::Matrix<double, 7, 1> xyz_uv_velocity;
			xyz_uv_velocity << x, y, z, 0.0, 0.0, 0.0, 0.0;
			featureFrame[feature_id].emplace_back(camera_id, xyz_uv_velocity);
		}

		prev_img = cur_img;
		prev_pts = cur_pts;
		prev_ipm = cur_ipm;
		prev_time = _cur_time;

		// save last camera-ground geometry for tracking
		prev_cg = cur_cg;

		return featureFrame;
	}

	map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> GroundTracker::trackImagePerspective(double _cur_time,
																								  const cv::Mat &_img, const double &threshold, const CameraGroundGeometry &cg,
																								  const Eigen::Matrix4d &Tckck_1, const bool &show_track)
	{
		cur_pts.clear();
		cur_cg = cg;

		cv::Mat img_undist = ipmproc->genUndistortedImage(_img);
		cv::Vec4d intrinsics = ipmproc->getUndistortedIntrinsics();

		cur_img = img_undist;

		cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
		clahe->apply(cur_img, cur_img);

		if (!(cur_cg == prev_cg))
		{
			ipmproc->updateCameraGroundGeometry(cur_cg);
		}

		mask = ipmproc->guessGroundROI();

		if (show_track)
			cv::imshow("mask", mask);

		if (!cur_img_semantic.empty())
		{
			vector<cv::Mat> img_channels_temp;
			cv::split(cur_img_semantic, img_channels_temp);
			cur_img_semantic = img_channels_temp[2];
			cur_img_semantic.setTo(255, cur_img_semantic == 7);
			cur_img_semantic.setTo(255, cur_img_semantic == 6);
			cur_img_semantic.setTo(255, cur_img_semantic == 20);
			mask.setTo(0, cur_img_semantic != 255);
		}

		// For visualization
		cv::Mat cur_img_show;
		cv::cvtColor(img_undist, cur_img_show, cv::COLOR_GRAY2BGR);

		vector<cv::Point2f> pred_pts;
		vector<uchar> status;
		vector<float> err;

		if (prev_pts.size() > 0)
		{
			Eigen::Matrix3d Rckck_1 = Tckck_1.block<3, 3>(0, 0);
			Eigen::Vector3d tckck_1 = Tckck_1.block<3, 1>(0, 3);

			// Predict tracked points
			for (int i = 0; i < prev_pts.size(); i++)
			{
				Eigen::Vector3d pc0k_1f = ipmproc->Perspective2Metric(prev_pts[i], &prev_cg);
				Eigen::Vector3d pc0kf = Rckck_1 * pc0k_1f + tckck_1;
				cv::Point2f uvc1kf = ipmproc->IPM2Perspective(ipmproc->Metric2IPM(pc0kf));
				pred_pts.push_back(uvc1kf);
			}
			cur_pts = pred_pts;
			// Optical flow tracking

			if (threshold > 10)
				cv::calcOpticalFlowPyrLK(prev_img, cur_img, prev_pts, cur_pts, status, err, cv::Size(KLT_PATCH_SIZE, KLT_PATCH_SIZE), 2,
										 cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01), cv::OPTFLOW_USE_INITIAL_FLOW);
			else
				cv::calcOpticalFlowPyrLK(prev_img, cur_img, prev_pts, cur_pts, status, err, cv::Size(KLT_PATCH_SIZE, KLT_PATCH_SIZE), 1,
										 cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01), cv::OPTFLOW_USE_INITIAL_FLOW);

			if (true) // flow back
			{
				vector<uchar> reverse_status;
				vector<cv::Point2f> reverse_pts = prev_pts;
				cv::calcOpticalFlowPyrLK(cur_img, prev_img, cur_pts, reverse_pts, reverse_status, err, cv::Size(KLT_PATCH_SIZE, KLT_PATCH_SIZE), 1,
										 cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01), cv::OPTFLOW_USE_INITIAL_FLOW);
				for (size_t i = 0; i < status.size(); i++)
				{
					if (status[i] && reverse_status[i] && distance(prev_pts[i], reverse_pts[i]) <= 0.5 && distance(cur_pts[i], pred_pts[i]) <= threshold)
					{
						status[i] = 1;
					}
					else
						status[i] = 0;
				}
			}

			reduceVector(prev_pts, status);
			reduceVector(cur_pts, status);
			reduceVector(ids, status);
			reduceVector(track_cnt, status);

			for (int i = 0; i < cur_pts.size(); i++)
			{
				cv::circle(cur_img_show, cur_pts[i], 0, cv::Scalar(0, 255, 0), 4);
				cv::line(cur_img_show, prev_pts[i], cur_pts[i], cv::Scalar(0, 255, 0), 1);
				cv::circle(mask, cur_pts[i], MIN_DIST, 0, -1);
			}

			if (cur_pts.size() >= 8)
			{
				printf("HM ransac begins\n");

				vector<uchar> mask_outlier;
				cv::Mat Hmatrix = cv::findHomography(cur_pts, prev_pts, mask_outlier, cv::RANSAC, 5.0);
				int size_a = cur_pts.size();
				printf("HM ransac: %d -> %lu: %f\n", size_a, cur_pts.size(), 1.0 * cur_pts.size() / size_a);

				reduceVector(prev_pts, mask_outlier);
				reduceVector(cur_pts, mask_outlier);
				reduceVector(ids, mask_outlier);
				reduceVector(track_cnt, mask_outlier);
			}
		}

		// Extract new points
		int n_max_cnt = MAX_CNT - static_cast<int>(cur_pts.size());
		vector<cv::Point2f> n_pts;
		if (n_max_cnt > 0)
		{
			if (mask.empty())
				cout << "mask is empty " << endl;
			if (mask.type() != CV_8UC1)
				cout << "mask type wrong " << endl;
			cv::goodFeaturesToTrack(cur_img, n_pts, MAX_CNT - cur_pts.size(), 0.001, MIN_DIST, mask, 3, true);
		}

		for (auto &p : n_pts)
		{
			n_pts.push_back(p);
			cur_pts.push_back(p);
			ids.push_back(n_id++);
			track_cnt.push_back(1);
			cv::circle(cur_img_show, p, 0, cv::Scalar(0, 0, 255), 4);
		}

		if (show_track)
		{
			cv::imshow("cur_img_show", cur_img_show);
			cv::waitKey(1);
		}

		map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> featureFrame;
		for (size_t i = 0; i < ids.size(); i++)
		{
			int feature_id = ids[i];
			cv::Point2f uvc0 = cur_pts[i];
			double x, y, z;
			x = (uvc0.x - intrinsics(2)) / intrinsics(0);
			y = (uvc0.y - intrinsics(3)) / intrinsics(1);
			z = 1;
			int camera_id = 0;

			Eigen::Matrix<double, 7, 1> xyz_uv_velocity;
			xyz_uv_velocity << x, y, z, 0.0, 0.0, 0.0, 0.0;
			featureFrame[feature_id].emplace_back(camera_id, xyz_uv_velocity);
		}

		prev_img = cur_img;
		prev_pts = cur_pts;
		prev_time = _cur_time;

		// save last camera-ground geometry for tracking
		prev_cg = cur_cg;

		return featureFrame;
	}

	bool GroundTracker::inBorder(const cv::Point2f &pt)
	{
		const int BORDER_SIZE = 1;
		int img_x = cvRound(pt.x);
		int img_y = cvRound(pt.y);
		return BORDER_SIZE <= img_x && img_x < col - BORDER_SIZE && BORDER_SIZE <= img_y && img_y < row - BORDER_SIZE;
	}

	double GroundTracker::distance(cv::Point2f &pt1, cv::Point2f &pt2)
	{
		// printf("pt1: %f %f pt2: %f %f\n", pt1.x, pt1.y, pt2.x, pt2.y);
		double dx = pt1.x - pt2.x;
		double dy = pt1.y - pt2.y;
		return sqrt(dx * dx + dy * dy);
	}

	map<int, vector<tuple<int, cv::Point2f>>> GroundTracker::feature2grid(vector<cv::Point2f> pts, vector<int> ids)
	{
		map<int, vector<tuple<int, cv::Point2f>>> grid_features;
		vector<vector<cv::KeyPoint>> new_feature_sieve(GRID_ROW * GRID_COL);
		for (int i = 0; i < pts.size(); i++)
		{
			int row = static_cast<int>(pts[i].y / GRID_HEIGHT);
			int col = static_cast<int>(pts[i].x / GRID_WIDTH);
			int code = row * GRID_COL + col;
			grid_features[code].push_back(make_tuple(ids[i], pts[i]));
		}
		return grid_features;
	}

	void GroundTracker::reduceVector(vector<cv::Point2f> &v, vector<uchar> status)
	{
		int j = 0;
		for (int i = 0; i < int(v.size()); i++)
			if (i < status.size() && status[i])
				v[j++] = v[i];
		v.resize(j);
	}

	void GroundTracker::reduceVector(vector<int> &v, vector<uchar> status)
	{
		int j = 0;
		for (int i = 0; i < int(v.size()); i++)
			if (status[i])
				v[j++] = v[i];
		v.resize(j);
	}
}