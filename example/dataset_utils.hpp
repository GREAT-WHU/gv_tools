#ifndef DATASET_UTILS_HPP
#define DATASET_UTILS_HPP

#include <string>
#include <map>
#include <Eigen/Dense>
#include <camodocal/gpl/gpl.h>
#include <fstream>

using namespace std;

map<double, string> load_camstamp(const string & filename)
{
	map<double, string> camstamp;
	string line;
	stringstream ss;
	ifstream ifs(filename);

	double sow; string imagename;
	while (ifs.good())
	{
		getline(ifs, line);
		if (line.size() < 1) continue;
		if (line[0] == '#') continue;
		for (int i = 0; i < line.size(); i++)
			if (line[i] == ',') line[i] = ' ';
		ss.clear(); ss.str("");
		ss << line;
		ss >> sow >> imagename;
		if(sow>1e9) sow/=1e9;
		camstamp[sow] = imagename;
	}

	return camstamp;
}

map<double, Eigen::Matrix4d> load_trajectory(const string & filename)
{
	map<double, Eigen::Matrix4d> traj;
	string line;
	stringstream ss;
	ifstream ifs(filename);

	double tt, x, y, z, qw, qx, qy, qz;
	while (ifs.good())
	{
		getline(ifs, line);
		if (line.size() < 1) continue;
		if (line[0] == '#') continue;
		ss.clear(); ss.str("");
		ss << line;
		ss >> tt >> x >> y >> z >> qw >> qx >> qy >> qz;
		if(tt>1e9) tt/=1e9;

		Eigen::Matrix4d pose = Eigen::Matrix4d::Identity();
		pose.block<3,3>(0,0) = Eigen::Quaterniond(qw,qx,qy,qz).toRotationMatrix();
		pose.block<3,1>(0,3) = Eigen::Vector3d(x,y,z);
		traj[tt] = pose;
	}
	return traj;
}

#endif