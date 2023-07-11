#pragma once

#include <Eigen/Eigen>
#include <map>

using namespace std;

double simple_pitch_compenstation(double tt, map<double, double> history_pitch, double duration = 2.0);
Eigen::VectorXd polyfit1d(Eigen::VectorXd x, Eigen::VectorXd y, int dim, Eigen::VectorXd weight = Eigen::VectorXd::Zero(0));
double modelPolyfit1d(Eigen::VectorXd param, double x);

/**
 * Simple implementation of temporal pitch compensation of camera-ground geometry
 * based on IMU attitudes. 
 * 
 * @param tt Current time.
 * @param history_pitch History IMU pitch estimates (in deg).
 * @param duration Time window considered for poly-fitting.
 * @return Pitch compenstation (in deg).
 */

double simple_pitch_compenstation(double tt, map<double, double> history_pitch, double duration)
{
    vector<double> pitch_window;
    vector<double> t_window;

    double t_start = (tt - duration);
    for (auto iter = history_pitch.rbegin(); iter != history_pitch.rend(); iter++)
    {
        if (tt - iter->first < duration)
        {
            t_window.push_back(iter->first - t_start);
            pitch_window.push_back(iter->second);
        }
    }
    if (pitch_window.size() < 10)
        return 0.0;
    else
    {
        Eigen::VectorXd t_v = Eigen::Map<Eigen::VectorXd>(t_window.data(), t_window.size());
        Eigen::VectorXd pitch_v = Eigen::Map<Eigen::VectorXd>(pitch_window.data(), pitch_window.size());
        Eigen::VectorXd param = polyfit1d(t_v, pitch_v, 2);
        double pitch_trend = modelPolyfit1d(param, tt - t_start);
        return pitch_trend - pitch_window.front();
    }
}

Eigen::VectorXd polyfit1d(Eigen::VectorXd x, Eigen::VectorXd y, int dim, Eigen::VectorXd weight)
{
    Eigen::VectorXd param = Eigen::VectorXd::Zero(dim + 1);
    Eigen::MatrixXd H(x.rows(), dim + 1);
    Eigen::VectorXd r(x.rows());
    Eigen::MatrixXd A(dim + 1, dim + 1);
    Eigen::VectorXd b(dim + 1);
    if (weight.rows() == 0)
    {
        weight = Eigen::VectorXd::Zero(x.rows());
        weight.setConstant(1.0);
    }
    for (int i_iter = 0; i_iter < 2; i_iter++)
    {
        H.setZero();
        r.setZero();
        for (int i = 0; i < x.size(); i++)
        {
            double y_est = 0.0;
            for (int j = 0; j < dim + 1; j++)
            {
                H(i, j) = pow(x(i), j);
                y_est += pow(x(i), j) * param(j);
            }
            r(i) = (y_est - y(i)) * weight(i);
            H.row(i) *= weight(i);

            // Hacky way to counter outliers.
            // TODO
            if (i_iter > 0)
            {
                double w;
                if (fabs(r(i)) < 0.05)
                    w = 1.0;
                else
                    w = 1 / (fabs(r(i)) / 0.05);
                r(i) *= w;
                H.row(i) *= w;
            }
        }
        A = H.transpose() * H;
        b = H.transpose() * r;
        param -= A.inverse() * b;
    }
    return param;
}

double modelPolyfit1d(Eigen::VectorXd param, double x)
{
    double y = 0.0;
    for (int j = 0; j < param.rows(); j++)
        y += param(j) * pow(x, j);
    return y;
}
