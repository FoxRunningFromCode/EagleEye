#include "EagleEye.h"

using namespace cv;
using namespace std;


float EagleEye::PixToMM(int pix) //converts 
{
    float pixels = 672.0f; //[pizels] calibration distance in pixels //før:!672
    float distance = 800.0f; //[mm] equals to:
    float calibraton = distance / pixels;
    return (pix * calibraton);
}

float EagleEye::MMtoPix(int mm) //converts 
{
    float pixels = 672.0f; //[pizels] calibration distance in pixels //før:!672
    float distance = 800.0f; //[mm] equals to:
    float calibraton = pixels / distance;

    return (mm * calibraton);
}



cv::Point3d EagleEye::FindBall(cv::Mat SCR, bool debug, cv::Point Center, int iLowH, int iHighH, int iLowS, int iHighS, int iLowV, int iHighV)
{ // Looks for an object within 250mm from the virtual centerpoint within certain color scale in RGB.
    using namespace cv;
    using namespace std;


    bool Debug_Active = debug;
    Mat Out_Image;
    Mat Edited_Image;
    Mat thresholdIMG;



    Edited_Image = SCR(Rect((Center.x - 210), (Center.y - 210), 420, 420));

    inRange(Edited_Image, Scalar(iLowH, iLowS, iLowV), Scalar(iHighH, iHighS, iHighV), thresholdIMG); //Threshold the image

    Mat kernel = getStructuringElement(MORPH_ELLIPSE, cv::Size(5, 5));

    erode(thresholdIMG, thresholdIMG, kernel, Point(-1, -1), 2);
    dilate(thresholdIMG, Out_Image, kernel, Point(-1, -1), 2);

    dilate(thresholdIMG, Out_Image, kernel, Point(-1, -1), 2);
    erode(thresholdIMG, thresholdIMG, kernel, Point(-1, -1), 2);

    Mat Detection_Image = Edited_Image;

    Moments m = moments(thresholdIMG, true);

    Point p((m.m10 / m.m00) + Center.x - 210, (m.m01 / m.m00) + Center.y - 210); //adds coords to the point as the picture for detecting is smaller than real picture.

    //ignore if there isnt an object.
    if (p.x < -1000)
    {
        p = Center;
    }

    Point Dist = p - Center;
    //convert pixel cords to 
    double floatX = PixToMM(Dist.x) * 0.988;
    double floatY = PixToMM(Dist.y) * 0.99;
    //cout << floatX<< " , "<<floatY;
    
    Point3d realToCenter(floatX, floatY, -20);



    //---------------------------------- Visual aid



    double deltaX = (PixToMM(abs(Dist.x)) * 0.988) * PixToMM((abs(Dist.x) * 0.988));
    double deltaY = (PixToMM(abs(Dist.y)) * 0.99) * PixToMM((abs(Dist.y) * 0.99));

    double DistanceCenter = (sqrt(deltaX + deltaY));

    line(SCR, p, Center, Scalar(255, 0, 0), 1);


    //Text for ball cords
    stringstream TempText;
    TempText << "(" << p.x << "," << p.y << ")";
    circle(Detection_Image, p, 23, Scalar(128, 0, 0), -1);
    circle(SCR, p, 23, Scalar(0, 0, 255), 2);


    putText(SCR, //target image
        TempText.str(), //text
        cv::Point(p.x - 80, p.y + 70), //top-left position
        cv::FONT_HERSHEY_DUPLEX,
        0.5,
        CV_RGB(255, 0, 0), //font color
        2);

    //text for distance

    stringstream Temp2Text;
    Temp2Text << "Distance [mm]: " << DistanceCenter << " , " << realToCenter;



    putText(SCR, //target image
        Temp2Text.str(), //text
        cv::Point(p.x - 80, p.y - 100), //top-left position
        cv::FONT_HERSHEY_DUPLEX,
        0.5,
        CV_RGB(255, 0, 0), //font color
        2);


    //-------------------------------- End visual aid






    //show you the image: to be placed at the end.
    if (Debug_Active)
    {
        system("CLS");
        std::cout << "---------------------------------------------------------------------------------------------------" << std::endl;
        std::cout << " Object found at [pix] : " << p << std::endl;
        std::cout << " Delta vector is calculated to be: [pix] " << Dist << std::endl;
        std::cout << " The real world coordinate is therfor [mm]: " << realToCenter << std::endl;
        std::cout << " Distance calculated to be: " << DistanceCenter << "MM" << std::endl;
        std::cout << "---------------------------------------------------------------------------------------------------" << std::endl;



        //namedWindow("Detected Balls", WINDOW_AUTOSIZE);
        //imshow("Detected Balls", Detection_Image);
        //namedWindow("SCR Image", WINDOW_AUTOSIZE);
        //imshow("SCR Image", SCR);
        namedWindow("Output Image", WINDOW_AUTOSIZE);
        imshow("Output Image", Out_Image);
        //namedWindow("Counting", WINDOW_AUTOSIZE);
        //imshow("Counting", dst);
        //namedWindow("Threshold2", WINDOW_AUTOSIZE);
        //imshow("Threshold2", Edited_Image);

        namedWindow("Threshold", WINDOW_AUTOSIZE);
        imshow("Threshold", thresholdIMG);
    }

    return realToCenter;
}

Mat EagleEye::Correct(Mat Input) //corrects for any lens distortion
{
    cv::Size frameSize(1440, 1080); //size of the input image.

    Vec<float, 5> k(-0.317761, 0.361848, 0, 0, 0); //values generated from another program

    cv::Matx33f K(1671.7371, 0, 719.5, 0, 1671.7371, 539.5, 0, 0, 1); //values generated from another program


    Mat mapX, mapY;
    initUndistortRectifyMap(K, k, cv::Matx33f::eye(), K, frameSize, CV_32FC1, mapX, mapY);

    Mat imgUndistorted;
    //Remap the image using the precomputed interpolation maps.
    remap(Input, imgUndistorted, mapX, mapY, cv::INTER_LINEAR);

    //Virtual center point:
    Point Center(759, 592); //note: we might be rotating around the wrong point. It should be center of image.
    Point RotateAroundP(Input.cols / 2, Input.rows / 2);
    //Rotate to fix poor mounting:
    double angle = -0.6;
    Mat rotation_matix = getRotationMatrix2D(Center, angle, 1.0);

    warpAffine(imgUndistorted, imgUndistorted, rotation_matix, imgUndistorted.size());
    return imgUndistorted;

}

cv::Point3d EagleEye::tableToRobot(cv::Point3d In)
{
    double xOffset = 238;
    double yOffset = -638;

    std::vector<double> homogen = {};
    std::vector<double> svar = {};
    double mellemregning = {};

    homogen.push_back(In.x);
    homogen.push_back(In.y);
    homogen.push_back(In.z);
    homogen.push_back(1);

    double mMatrix[4][4] = { {-0.9239, -0.3827,  0, xOffset},
                             {-0.3827,  0.9239,  0, yOffset},
                             {      0,       0, -1,    0},
                             {      0,       0,  0,    1} };


    for (size_t i = 0; i < 4; i++)
    {
        for (size_t j = 0; j < 4; j++)
        {
            mellemregning += mMatrix[i][j] * homogen[j];
        }
        // std::cout << mellemregning << " ";
        svar.push_back(mellemregning);
        mellemregning = 0;
    }

    cv::Point3d returnSvar(svar[0], svar[1], svar[2]);

    //std::cout << returnSvar.x << " " << returnSvar.y << std::endl;
    return returnSvar;
}

cv::Point3d EagleEye::grabObject(int mItem) {

    Point3d robotBallPos;
    Mat Image; //loads image for usage
    Mat Undistorted_Image;
    int myExposure = 20000;

    // The exit code of the sample application.
    int exitCode = 0;

    // Automagically call PylonInitialize and PylonTerminate to ensure the pylon runtime system
    // is initialized during the lifetime of this object.
    Pylon::PylonAutoInitTerm autoInitTerm;

    try
    {
        // Create an instant camera object with the camera device found first.
        Pylon::CInstantCamera camera(Pylon::CTlFactory::GetInstance().CreateFirstDevice());

        // Get a camera nodemap in order to access camera parameters.
        GenApi::INodeMap& nodemap = camera.GetNodeMap();

        // Open the camera before accessing any parameters.
        camera.Open();
        // Create pointers to access the camera Width and Height parameters.
        GenApi::CIntegerPtr width = nodemap.GetNode("Width");
        GenApi::CIntegerPtr height = nodemap.GetNode("Height");
        Pylon::CEnumParameter(nodemap, "LightSourcePreset").SetValue("Daylight6500K");
        Pylon::CIntegerParameter(nodemap, "Width").SetValue(1440); // max 1448 x 1084
        Pylon::CIntegerParameter(nodemap, "Height").SetValue(1080);
        Pylon::CIntegerParameter(nodemap, "OffsetX").SetValue(8); //324
        Pylon::CIntegerParameter(nodemap, "OffsetY").SetValue(4);
        Pylon::CBooleanParameter(nodemap, "AcquisitionFrameRateEnable").SetValue(false);
        Pylon::CFloatParameter(nodemap, "ExposureTime").SetValue(20000.0);
        // The parameter MaxNumBuffer can be used to control the count of buffers
        // allocated for grabbing. The default value of this parameter is 10.
        //camera.MaxNumBuffer = 5;

        // Create a pylon ImageFormatConverter object.
        Pylon::CImageFormatConverter formatConverter;
        // Specify the output pixel format.
        formatConverter.OutputPixelFormat = Pylon::PixelType_BGR8packed;
        // Create a PylonImage that will be used to create OpenCV images later.
        Pylon::CPylonImage pylonImage;

        // Create an OpenCV image.
        cv::Mat openCvImage;


        // Set exposure to manual
        GenApi::CEnumerationPtr exposureAuto(nodemap.GetNode("ExposureAuto"));
        if (GenApi::IsWritable(exposureAuto)) {
            exposureAuto->FromString("Off");
            std::cout << "Exposure auto disabled." << std::endl;
        }

        // Set custom exposure
        GenApi::CFloatPtr exposureTime = nodemap.GetNode("ExposureTime");
        std::cout << "Old exposure: " << exposureTime->GetValue() << std::endl;
        if (exposureTime.IsValid()) {
            if (myExposure >= exposureTime->GetMin() && myExposure <= exposureTime->GetMax()) {
                exposureTime->SetValue(myExposure);
            }
            else {
                exposureTime->SetValue(exposureTime->GetMin());
                std::cout << ">> Exposure has been set with the minimum available value." << std::endl;
                std::cout << ">> The available exposure range is [" << exposureTime->GetMin() << " - " << exposureTime->GetMax() << "] (us)" << std::endl;
            }
        }
        else {

            std::cout << ">> Failed to set exposure value." << std::endl;
            return cv::Point3d(0,0,100);
        }
        std::cout << "New exposure: " << exposureTime->GetValue() << std::endl;

        // Start the grabbing of c_countOfImagesToGrab images.
        // The camera device is parameterized with a default configuration which
        // sets up free-running continuous acquisition.
        camera.StartGrabbing(Pylon::GrabStrategy_LatestImageOnly);

        // This smart pointer will receive the grab result data.
        Pylon::CGrabResultPtr ptrGrabResult;
        //---------------------------------------------------------------------------------------------
        
        int input = mItem;
        


        int iLowH = 0;
        int iHighH = 60;

        int iLowS = 0;
        int iHighS = 132;

        int iLowV = 45;
        int iHighV = 168;



        switch (input) {
        case 1: //case Blue (da ba dee, da ba dei)
            iLowH = 0;
            iHighH = 160;

            iLowS = 0;
            iHighS = 151;

            iLowV = 0;
            iHighV = 255;
            break;
        case 2: //case orange //Remember BGR
            iLowH = -1; //blue 0
            iHighH = 60;//60

            iLowS = 120; //green 120
            iHighS = 255; //230

            iLowV = 60; //red 60
            iHighV = 256; // 255
            break;
        case 3:  //case Låg
            iLowH = 21; //blue
            iHighH = 114;

            iLowS = 57; //green
            iHighS = 123;

            iLowV = 65; // Red
            iHighV = 166;
            break;
        case 4: //case dynamic
            iLowH = 255;
            iHighH = 255;

            iLowS = 255;
            iHighS = 255;

            iLowV = 255;
            iHighV = 255;

            namedWindow("Control", WINDOW_AUTOSIZE); //create a window called "Control"
            createTrackbar("BlueLow", "Control", &iLowH, 255); //Hue (0 - 179)
            createTrackbar("BlueHigh", "Control", &iHighH, 255);

            createTrackbar("GreenLow", "Control", &iLowS, 255); //Saturation (0 - 255)
            createTrackbar("GreenHigh", "Control", &iHighS, 255);

            createTrackbar("RedLow", "Control", &iLowV, 255);//Value (0 - 255)
            createTrackbar("RedHigh", "Control", &iHighV, 255);
            break;
        }

        //-----------------------------------------------------------------------------------



        // image grabbing loop
        int frame = 1;
        //while (camera.IsGrabbing())
        //{
            // Wait for an image and then retrieve it. A timeout of 5000 ms is used.
            camera.RetrieveResult(5000, ptrGrabResult, Pylon::TimeoutHandling_ThrowException);

            // Image grabbed successfully?
            if (ptrGrabResult->GrabSucceeded())
            {

                // Convert the grabbed buffer to a pylon image.
                formatConverter.Convert(pylonImage, ptrGrabResult);

                // Create an OpenCV image from a pylon image.
                openCvImage = cv::Mat(ptrGrabResult->GetHeight(), ptrGrabResult->GetWidth(), CV_8UC3, (uint8_t*)pylonImage.GetBuffer());



                //////////////////////////////////////////////////////
                //////////// Here your code begins ///////////////////
                //////////////////////////////////////////////////////


                //-----------------------------------------------------------------------------

                //center point used for graphical nonsense. AND input for findball() its the refference point for distance.
                Point Center(759, 592);


                Mat Corrected_Image;
                Corrected_Image = Correct(openCvImage);

                Point3d BallPos = FindBall(Corrected_Image, true, Center, iLowH, iHighH, iLowS, iHighS, iLowV, iHighV);
                robotBallPos = tableToRobot(BallPos);

                //debugging Crosshair
                if (1)
                {

                    int TestpointX = MMtoPix(250);
                    int TestpointY = MMtoPix(250);

                    //create points for the cross
                    Point New(Center.x + TestpointX, Center.y + TestpointY);
                    Point NewU(Center.x + TestpointX, Center.y);
                    Point NewD(Center.x - TestpointX, Center.y);
                    Point NewL(Center.x, Center.y + TestpointY);
                    Point NewR(Center.x, Center.y - TestpointY);

                    //draw Cross
                    circle(Corrected_Image, Center, 4, Scalar(255, 0, 0), 1);
                    circle(Corrected_Image, NewU, 3, Scalar(100, 255, 0), 2);
                    circle(Corrected_Image, NewD, 3, Scalar(100, 255, 0), 2);
                    circle(Corrected_Image, NewL, 3, Scalar(100, 255, 0), 2);
                    circle(Corrected_Image, NewR, 3, Scalar(100, 255, 0), 2);

                    //text for detected object in robot frame coords

                    stringstream Temp2Text;
                    Temp2Text << "X: " << robotBallPos.x << " , Y:  " << robotBallPos.y << " , Z:  " << robotBallPos.z << "\n";

                    putText(Corrected_Image, //target image
                        Temp2Text.str(), //text
                        cv::Point(10, 50),
                        cv::FONT_HERSHEY_DUPLEX,
                        0.7,
                        CV_RGB(255, 0, 0), //font color
                        2);

                }




                namedWindow("Output Image", WINDOW_AUTOSIZE);
                imshow("Output Image", Corrected_Image);



                // Detect key press and quit if 'q' is pressed
                int keyPressed = cv::waitKey(1);
                if (keyPressed == 'q') { //quit
                    std::cout << "Shutting down camera..." << std::endl;
                    camera.Close();
                    std::cout << "Camera successfully closed." << std::endl;
                    return cv::Point3d(0, 0, 100);
                }
                else if (keyPressed == 's') {
                    cv::imwrite("img" + std::to_string(frame) + ".png", openCvImage);

                }




                ////////////////////////////////////////////////////
                //////////// Here your code ends ///////////////////
                ////////////////////////////////////////////////////

                frame++;
            }
            else
            {
                std::cout << "Error: " << ptrGrabResult->GetErrorCode() << " " << ptrGrabResult->GetErrorDescription() << std::endl;
            }
        }

   // }
    catch (GenICam::GenericException& e)
    {
        // Error handling.
        std::cerr << "An exception occurred." << std::endl
            << e.GetDescription() << std::endl;
        exitCode = 1;
    }

    return robotBallPos;




}