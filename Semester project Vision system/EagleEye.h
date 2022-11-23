#pragma once
#include <iostream>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <stdio.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <cmath>
#include <pylon/PylonIncludes.h>
#include <string>
#include <GenApi/GenApi.h>


class EagleEye
{
public:
	float PixToMM(int pix);
	float MMtoPix(int mm);

	cv::Point3d FindBall(cv::Mat SCR, bool debug, cv::Point Center, int iLowH, int iHighH, int iLowS, int iHighS, int iLowV, int iHighV);
	cv::Mat Correct(cv::Mat Input);

	cv::Point3d tableToRobot(cv::Point3d In);

	cv::Point3d grabObject(int mItem);
};

