#include "ground_tracker.h"
#include "dataset_utils.hpp"
#include "simple_compensation.hpp"

#ifdef DEBUG_PITCH_COMPENSATION
	std::ofstream ofs_debug_pitch("pitch.log");
#endif

int main(int argc,char *argv[])
{
	if(argc<5 || argc>5)
	{
        cerr << endl
             << "Usage: track_dataset path_yaml path_dataset path_stamp path_odom"
             << endl;
        return 1;
	}
	string path_yaml(argv[1]);
	string path_dataset(argv[2]);
	string path_stamp(argv[3]);
	string path_odom(argv[4]);


	gv::CameraConfig config(path_yaml);
	gv::GroundTracker traker(config);

	cv::FileStorage fsSettings(path_yaml, cv::FileStorage::READ);
    bool enable_pitch_comp = (int)fsSettings["enable_pitch_comp"];
    double pitch_comp_windowsize = fsSettings["pitch_comp_windowsize"];
    fsSettings.release();

	std::map<double, double> all_imu_pitch;

	auto traj = load_trajectory(path_odom);
	auto stamp = load_camstamp(path_stamp);

	printf("odometry: %20.4f~%20.4f\n",traj.begin()->first,traj.rbegin()->first);
	printf("camstamp: %20.4f~%20.4f\n",stamp.begin()->first,stamp.rbegin()->first);

	Eigen::Matrix4d last_pose = Eigen::Matrix4d::Identity();
	
	for (auto stamp_iter = stamp.begin(); stamp_iter != stamp.end(); stamp_iter++)
	{
		double tt = stamp_iter->first;
		auto pose_iter = traj.lower_bound(tt - 1e-3);
		if (fabs(pose_iter->first - tt) > 1e-2) continue;

		// Using relative pose prediction to assist feature tracking 
		// (using GT poses in this example, but could be obtained via 
		// real-time estimation of VIO/VINS)
		Eigen::Matrix4d Twik_1 = last_pose;
		Eigen::Matrix4d Twik = pose_iter->second;
		Eigen::Matrix4d Tckck_1 = (Twik * config.Tic).inverse() * (Twik_1 * config.Tic);
		
		// Here we use pre-calibrated camera-ground geometry. While 
		// this could be online estimated and continuously refined 
		// in a VIO estimator.
 		gv::CameraGroundGeometry cg = config.cg;

		// Calculate temporary camera-ground geometry considering 
		// high-frequency pitch compenstation.
		if(enable_pitch_comp)
		{
			all_imu_pitch[tt] = gv::m2att(Twik.block<3,3>(0,0)).x() * 180 / M_PI;
			double comp = simple_pitch_compenstation(tt, all_imu_pitch, pitch_comp_windowsize);
			cg.update(cg.getAlpha(), cg.getTheta()+comp);
			
#ifdef DEBUG_PITCH_COMPENSATION
			ofs_debug_pitch<<setprecision(4)<<setiosflags(ios::fixed)<<tt
			<<all_imu_pitch[tt]<<" "<<comp<<" "<< <<std::endl;
#endif
		}

		// main feature tracking
		cv::Mat img = cv::imread(path_dataset +'/'+ stamp_iter->second);
		cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
		traker.trackImage(tt, img, 10.0, cg, Tckck_1,true);
		// traker.trackImagePerspective(tt, img, 10.0, cg, Tckck_1,true);

		last_pose = pose_iter->second;

	}
	return 0;
}

