// VolleyballTrace.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"
#include "CameraCalib.h"
#include "HarrisDetector.h"

bool draw_flag;
CvRect box;


void draw_box(IplImage* image,CvRect box)
{
	cvRectangle(image,Point(box.x,box.y),Point(box.x + box.width,box.y + box.height),CV_RGB(255,0,0),1);
}

void onMouse(int event,int x,int y,int flags,void* param)
{
	IplImage* image = (IplImage*) param;
	switch(event)
	{
	case CV_EVENT_MOUSEMOVE:
		{
			if(draw_flag)
			{
				box.width = x - box.x;
				box.height = y - box.y;
			}
		}
		break;
	case CV_EVENT_LBUTTONDOWN:
		{
			draw_flag  = true;
			box = cvRect(x,y,0,0);
		}
		break;
	case CV_EVENT_LBUTTONUP:
		{
			draw_flag = false;
			if(box.width < 0)
			{
				box.x = box.x + box.width;
				box.width *= -1;
			}
			if(box.height < 0)
			{
				box.y += box.height;
				box.height *= -1;
			}
			draw_box(image,box);
		}
		break;
	}
}
int InitDetector(HarrisDetector& harris)
{
	CvCapture *capture;
	IplImage* img;
	capture = cvCaptureFromCAM(1);
	cvNamedWindow("ROI",1);
	while(true)
	{
		img = cvQueryFrame(capture);
		cvShowImage("ROI",img);
		if(cvWaitKey(10) >= 0)
		{
			break;
		}
	}
	IplImage* originImage = cvCloneImage(img);
	IplImage* temp = cvCloneImage(img);
	cvSetMouseCallback("ROI",onMouse,(void*)img);
	while(1){
	   cvCopy(img,temp);
       if(draw_flag) draw_box(temp,box);
       cvShowImage("ROI",temp);
		if(cvWaitKey(15) >= 0) break;
	}


//	cvReleaseImage(&temp);
//	cvReleaseImage(&img);
//	cvReleaseCapture(&capture);
	cvDestroyWindow("ROI");

	cvSetImageROI(originImage,box);
	cvSaveImage("data/image/ROI.jpg",originImage);
	cv::Mat image = originImage;
	cv::Mat imageGray;
    cv::cvtColor(image,imageGray,CV_BGR2GRAY);
	harris.detect(imageGray);
	vector<cv::Point> pts;
	harris.getCorners(pts,0.01);
	harris.drawOnImage(image,pts);
	namedWindow("result",1);
	imshow("result",image);
	return 0;
}

int main(int argc, _TCHAR* argv[])
{
	//camera calibration
	CameraCalib camera;
	CameraCalib::CornerDatas cornerDatas;
	CameraCalib::CameraParams cameraParams;
	CameraCalib::RemapMatrixs remapMatrixs;
	Calib(camera,cornerDatas,cameraParams,remapMatrixs);

	//Initial Corner
	HarrisDetector harris;
	InitDetector(harris);

	//define the variables which used to match
	Mat descriptor = harris.getHarris();
	vector<cv::KeyPoint> pts;
	harris.getCorners(pts,0.01);
	IplImage* frame;
	CvCapture* capture;
	capture = cvCaptureFromCAM(1);
	namedWindow("Trace",1);

	//initial Kalman filter
	Point center = Point((int)(box.x + box.width/2),(int)(box.y + box.height/2));
	const int stateNum = 4;
	const int measureNum = 2;
	CvKalman* kalman = cvCreateKalman(stateNum,measureNum,0);
	CvMat* process_noise = cvCreateMat(stateNum,1,CV_32FC1);
	CvMat* measurement = cvCreateMat(measureNum,1,CV_32FC1);
	float A[stateNum][stateNum] = {
		1,0,1,0,
		0,1,0,1,
		0,0,1,0,
		0,0,0,1
	};
	memcpy( kalman->transition_matrix->data.fl,A,sizeof(A));
	cvSetIdentity(kalman->measurement_matrix,cvRealScalar(1) );
	cvSetIdentity(kalman->process_noise_cov,cvRealScalar(1e-5));
	cvSetIdentity(kalman->measurement_noise_cov,cvRealScalar(1e-1));
	cvSetIdentity(kalman->error_cov_post,cvRealScalar(1));
	kalman->state_post->data.fl[0] = (float)center.x;
	kalman->state_post->data.fl[1] = (float)center.y;
	kalman->state_post->data.fl[2] = 0;
	kalman->state_post->data.fl[3] = 0;
	Rect trackWindow;

	while(1)
	{
		frame = cvQueryFrame(capture);
		trackWindow = KalmanPre(kalman,box);
		//add Kalman code and calculate the position of the box
		Point centerPosition = Match(descriptor,pts,box,trackWindow,frame);
		Rect newBox = Rect(centerPosition.x - box.width/2,centerPosition.y - box.height/2,box.width,box.height);
		draw_box(frame,newBox);
		cvShowImage("Trace",frame);
		measurement->data.fl[0] = (float)centerPosition.x;
		measurement ->data.fl[1] = (float)centerPosition.y;
		KalmanUpdate(kalman,measurement);
		if(waitKey(10) == 27)
			break;
		//Match(descriptor,pts,box,frame);
		
	}
//	cvReleaseCapture(&capture);
//	cvReleaseImage(&frame);
	cvDestroyWindow("Trace");
	waitKey(0);
	return 0;
}

