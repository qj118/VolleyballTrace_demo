#include <opencv2\highgui\highgui.hpp>
#include <opencv2\opencv.hpp>
#include <opencv2\core\core_c.h>
#include <opencv2\core\core.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include <vector>
using namespace std;
using namespace cv;

class HarrisDetector{
private:
	cv::Mat cornerStrength;
	cv::Mat cornerTh;
	cv::Mat localMax;
	int neighbourhood;
	int aperture;
	double k;
	double maxStrength;
	double threshold;
	int nonMaxSize;
	cv::Mat kernel;
public:
	HarrisDetector():neighbourhood(3),aperture(3),k(0.05),maxStrength(0.0),threshold(0.01),nonMaxSize(3){
		setLocalMaxWindowSize(nonMaxSize);
	};
	void setLocalMaxWindowSize(int nonMaxSize)
	{
		kernel = nonMaxSize;
	};
	void detect(const cv::Mat& image);
	cv::Mat getCornerMap(double qualityLevel);
	void getCorners(vector<cv::Point>& points,double qualityLevel);
	void getCorners(vector<cv::Point>& points,const cv::Mat& cornerMap);
	void getCorners(vector<cv::KeyPoint>& points,double qualityLevel);
	void getCorners(vector<cv::KeyPoint>& points,const cv::Mat& cornerMap);
	void drawOnImage(cv::Mat& image,const vector<cv::Point>& points,cv::Scalar color = cv::Scalar(255,255,255),int radius = 3,int thickness = 2);
	cv::Mat getHarris(){
		return cornerStrength;
	}
};
Point Match(const Mat& descriptor,const vector<cv::KeyPoint>& pts,const CvRect& box,const CvRect& trackWindow,IplImage* frame);
Rect KalmanPre(CvKalman* kalman,const Rect& box);
int KalmanUpdate(CvKalman* kalman,CvMat* measurement);