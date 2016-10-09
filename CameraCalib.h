#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/core/core_c.h>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

class CameraCalib
{
public:
	CameraCalib(void);
	~CameraCalib(void);
	struct CornerDatas
	{
		int nPoints;
		int nImages;
		int nPointsPerImage;
		cv::Size imageSize;
		cv::Size boardSize;
		vector<vector<cv::Point3f> > objectPoints;
		vector<vector<cv::Point2f> > imagePoints;
	};
	struct CameraParams
	{
		cv::Size imageSize;
		cv::Mat cameraMatrix;
		cv::Mat distortionCoefficients;
		vector<cv::Mat> rotations;
		vector<cv::Mat> translations;
		int flags;
	};
	struct RemapMatrixs
	{
		cv::Mat mX;
		cv::Mat mY;
		cv::Rect roi;
	};

	void setWorkDir(const char* workDir){m_workDir = workDir;}
	int initCornerData(int nImages,cv::Size imageSize,cv::Size boardSize,float squareWidth,CornerDatas& cornerDatas);
	int resizeCornerData(int nImages,CornerDatas& cornerDatas);
	int loadCornerData(const char* filename,CornerDatas& cornerDatas);
	int saveCornerData(const char* filename,const CornerDatas& cornerDatas);
	int detectCorners(cv::Mat& img, CornerDatas& cornerDatas,int imageCount);
	int loadCameraParams(const char* filename,CameraParams& cameraParams);
	int saveCameraParams(const CameraParams& cameraParams,const char* filename );
	int calibrateSingleCamera(CornerDatas& cornerDatas,CameraParams& cameraParams);
	int getCameraCalibrateError(vector<vector<cv::Point3f> >& _objectPoints,vector<vector<cv::Point2f> >& _imagePoints,CameraParams& cameraParams,double& err);
	int rectifySingleCamera(CameraParams& cameraParams,RemapMatrixs& remapMatrixs);
	int remapImage(cv::Mat& img,cv::Mat& imgout,RemapMatrixs& remapMatrixs);
	void showText(cv::Mat& img,string text);
public:
	string m_workDir;
};
int Calib(CameraCalib& camera,CameraCalib::CornerDatas& cornerDatas,CameraCalib::CameraParams& cameraParams,CameraCalib::RemapMatrixs& remapMatrixs);