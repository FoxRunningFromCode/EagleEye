#include "EagleEye.h"



int main() 
{
	EagleEye Vision;
	cv::Point3d Object;
	Object = Vision.grabObject(2);
	std::cout << Object.x << " , " << Object.y << " , " << Object.z << std::endl;

	return 1;
}