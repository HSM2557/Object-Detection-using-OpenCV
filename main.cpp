#include "opencv2/imgcodecs.hpp" 
#include "opencv2/highgui/highgui.hpp" 
#include "opencv2/imgproc/imgproc.hpp" 
#include <opencv2/opencv.hpp> 
#include "raspicam/raspicam_cv.h" 
 
#define UNIT_PIXEL_W 0.0008234375 
#define UNIT_PIXEL_H 0.000825 
 
struct sColorDetectorPar 
{	int hLow;   
	int hHigh;     
	int sLow;     
	int sHigh;     
	int vLow;     
	int vHigh; 
 
    sColorDetectorPar() 
    {         
		hLow  = 26;         
		hHigh = 60;         
		sLow  = 68;         
		sHigh = 255;         
		vLow  = 51;         
		vHigh = 255;
    } 
}; 
 
struct sDistanceDetectorPar 
{   int w;     
	int h;     
	sDistanceDetectorPar() 
    {         
		w = 76;         
		h = 76; 
    } 
}; 
 
 
using namespace std; 
using namespace cv; 
 
int threshold_value = 90;
Mat image,image1; 
Mat grayBlurred,grayImage;
Mat cannyOut; 
cv::Mat blobsColor; 
cv::Mat blobsCounter; 
 
vector<vector<Point> > contours; //edge is a series of pixels 
 
void canny(int,void*); void ContourDetect(cv::Mat bgr,cv::Mat& blobsCounter,char dbgFlag); 
void DistanceDetect(cv::Mat bgr,const sDistanceDetectorPar par);
void CooridinateDetect(cv::Mat bgr); 
void detectColor(cv::Mat bgr, const sColorDetectorPar par, cv::Mat& blobsColor); 
int main() 
{     
	
	time_t timer_begin,timer_end; 
 
    raspicam::RaspiCam_Cv Camera; 
 
    Camera.set( CV_CAP_PROP_FORMAT, CV_8UC3 ); // gray: CV_8UC1; color: CV_8UC3 
    Camera.set( CV_CAP_PROP_FRAME_HEIGHT, 480 ); // 960 
    Camera.set( CV_CAP_PROP_FRAME_WIDTH, 640 ); // 1280 
    Camera.set( CV_CAP_PROP_EXPOSURE, -1); 
 
 
    //Open camera
                                                                                                                                                                                                                                                                                                   cout << "Opening Camera..." << endl;     if (!Camera.open())
    { 
        cerr << "Error opening the camera" << endl;         
		cv::waitKey();        
		return -1; 
    } 
    int fps = Camera.get( CV_CAP_PROP_FPS ); 
 
    sColorDetectorPar parColorDetector;    
	namedWindow("Color Detector", WINDOW_NORMAL);    
	resizeWindow("Color Detector", 400, 280);    
	moveWindow("Color Detector", 880,300); 
    createTrackbar("LowH", "Color Detector", &parColorDetector.hLow, 179); //Hue (0 - 179)
                                                                                                                                                                                                                                                                                                   createTrackbar("HighH", "Color Detector", &parColorDetector.hHigh, 179);
    createTrackbar("LowS", "Color Detector", &parColorDetector.sLow, 255); 
//Saturation (0 - 255)
                                                                                                                                                                                                                                                                                                   createTrackbar("HighS", "Color Detector", &parColorDetector.sHigh, 255);
    createTrackbar("LowV", "Color Detector", &parColorDetector.vLow, 255);//Value (0 - 255)
                                                                                                                                                                                                                                                                                                   createTrackbar("HighV", "Color Detector", &parColorDetector.vHigh, 255);
 
    sDistanceDetectorPar parDistanceDetector;     
	namedWindow("Distance Detector", CV_WINDOW_NORMAL);    
	resizeWindow("Distance Detector", 580, 150);    
	moveWindow("Distance Detector",300,430);     
	createTrackbar("Width","Distance Detector",&parDistanceDetector.w, 300);     
	createTrackbar("Height", "Distance Detector",&parDistanceDetector.h, 300); 
 
    //Start capture
                                                                                                                                                                                                                                                                                                   time ( &timer_begin ); int i=0;     for (;;)     {         i++;
        Camera.grab(); 
        Camera.retrieve (image);         
		namedWindow("live image", WINDOW_NORMAL);         
		resizeWindow("live image", 300, 280);         
		moveWindow("live image", 0,0);         
		cv::imshow("live image", image); 
 
        //HSV
        cvtColor(image, image1, CV_BGR2HSV);
 
        //gray 
        cvtColor(image,grayImage,CV_BGR2GRAY);         
		namedWindow("gray", WINDOW_NORMAL);         
		resizeWindow("gray", 300, 280);         
		moveWindow("gray", 0,280);         
		imshow("gray",grayImage); 
 
        //eliminate noise 
        GaussianBlur(grayImage, grayBlurred,Size(5,5),0); 
 
        //namedWindow("Threshold",WINDOW_AUTOSIZE); 
        //createTrackbar("threshold","Threshold",&threshold_value,255,canny); 
        //canny(threshold_value,0); 
 
        detectColor(image, parColorDetector, blobsColor); 
        ContourDetect(image,blobsCounter, 1); 
        DistanceDetect(image, parDistanceDetector); 
        CooridinateDetect(image); 
 
        namedWindow("final", WINDOW_NORMAL);         
		resizeWindow("final", 580, 470);         
		moveWindow("final", 300,0);         
		imshow("final",image); 
 
 
        if ( i%5==0 ) 
        { 
            cout << "\r captured " << i <<" images" << ", size:" << image.rows << "x" << image.cols << ", FPS: " << fps << std::flush; 
        }        
		if (cv::waitKey(5) > 0) 
		{    
			break;
        } 
    } 
 
    // Release camera 
    Camera.release(); 
 
    //show time statistics
                                                                                                                                                                                                                                                                                                   time ( &timer_end ); /* get current time; same as: timer = time(NULL)  */     double secondsElapsed = difftime ( timer_end,timer_begin );
    std::cout << secondsElapsed <<" seconds for " << i << "  frames : FPS = "<<  ( float ) ( ( float ) ( i ) /secondsElapsed ) << endl;
 
    //save image
                                                                                                                                                                                                                                                                                                   cv::imwrite("raspicam_cv_image.jpg",image); cout << "Image saved at raspicam_cv_image.jpg" << endl;
     return 0; 
} 
 
void canny(int,void*) 
{ 
    Canny(grayBlurred,cannyOut,threshold_value, 180);//edge extraction and image binaryzation     imshow("binary",cannyOut); 
} 
 
void detectColor(cv::Mat bgr, const sColorDetectorPar par, cv::Mat& blobsColor) 
{ 
    Mat imgHSV; 
    cvtColor(bgr, imgHSV, cv::COLOR_BGR2HSV); //Convert the captured frame from BGR to HSV 
 
    /// Threshold HSV image 
    inRange(imgHSV, cv::Scalar(par.hLow, par.sLow, par.vLow), cv::Scalar(par.hHigh, par.sHigh, par.vHigh), blobsColor); 
 
 
    // morphological opening (removes small objects from the foreground) 
    cv::erode(blobsColor, blobsColor, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5)) ); 
    cv::dilate( blobsColor, blobsColor, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5)) ); 
 
    /// morphological closing (removes small holes from the foreground) 
    cv::dilate( blobsColor, blobsColor, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5)) ); 
    cv::erode(blobsColor, blobsColor, cv::getStructuringElement(cv::MORPH_ELLIPSE, 
    cv::Size(5, 5)) ); 
    namedWindow("Color Detection", WINDOW_NORMAL);     
	resizeWindow("Color Detection", 400, 350);     
	moveWindow("Color Detection", 880,0);     
	imshow("Color Detection", blobsColor); 
} 
 
 
void ContourDetect(cv::Mat bgr,cv::Mat& blobsCounter,char dbgFlag) 
{ 
    findContours(blobsColor, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, Point(0,0));     for(int i = 0; i < contours.size(); i++) 

    {   

		drawContours(blobsCounter, contours, i, Scalar(255), 2, 8, -1, 0, Point(0,0));

    } 
    if (dbgFlag == 0x01) 
    {    for(int i = 0; i < contours.size(); i++) 
        {  

			drawContours(bgr, contours, i, Scalar(0,255,0), 2, 8, -1, 0, Point(0,0)); 

        }
	} 
} 
 
 
void DistanceDetect(cv::Mat bgr,const sDistanceDetectorPar par) 
{     vector<cv::Point> maxAreaContour;     const double f = 0.35; 
double maxArea = 0;     for (int i = 0; i < contours.size(); i++) 
    {         
		double area = fabs(cv::contourArea(contours[i]));         if (area > maxArea) 
        {             
			maxArea = area;             maxAreaContour  = contours[i]; 
        } 
    } 
 
    // Take the rect
  cv::Rect rect = cv::boundingRect(maxAreaContour);
 
    // Calculate rectangle size
    double width = rect.width * UNIT_PIXEL_W;     
	double height = rect.height * UNIT_PIXEL_H;
 
    // calculate distance
      double distanceW = par.w * f / width / 10;     
	  double distanceH = par.h * f / height / 10;
 
    char disW[50];     
	sprintf(disW, "DistanceW : %.2fcm", distanceW);     
	char disH[50];    
	sprintf(disH, "DistanceH : %.2fcm", distanceH); 
    cv::putText(bgr, disW, cv::Point(5, 20), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(0, 255, 0), 1, 8); 
    cv::putText(bgr, disH, cv::Point(5, 40), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(0, 255, 0), 1, 8); 
 
} 
 
 
void CooridinateDetect(cv::Mat bgr) 
{	vector<Moments> mu(contours.size() );    
	vector<Point2f> mc(contours.size() );    
	for(int i = 0; i < contours.size(); i++) 
    {         
		mu[i] = moments( contours[i], false );        
		mc[i] = Point2f( mu[i].m10/mu[i].m00 , mu[i].m01/mu[i].m00 ); 
        circle(bgr, mc[i], 4, Scalar(0, 0, 255), -1, 8, 0 ); 
        float x = mc[i].x;         
		float y = mc[i].y; 
        char Position[50];         
		sprintf(Position, "Position [%.2f,%.2f]", x,y); 
        cv::putText(bgr, Position, cv::Point(5, 60), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(0, 255, 0), 1, 8); 
    } 
} 
