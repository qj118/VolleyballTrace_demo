#include "stdafx.h"
#include "function.h"
#include "HarrisDetector.h"


void HarrisDetector::detect(const cv::Mat& image){
	cv::cornerHarris(image,cornerStrength,neighbourhood,aperture,k);
	double minStrength;
	cv::minMaxLoc(cornerStrength,&minStrength,&maxStrength);
	cv::Mat dilated;
	cv::dilate(cornerStrength,dilated,cv::Mat());
	cv::compare(cornerStrength,dilated,localMax,cv::CMP_EQ);
}

cv::Mat HarrisDetector::getCornerMap(double qualityLevel)
{
	cv::Mat cornerMap;
	threshold = qualityLevel * maxStrength;
	cv::threshold(cornerStrength,cornerTh,threshold,255,cv::THRESH_BINARY);
	cornerTh.convertTo(cornerMap,CV_8U);
	cv::bitwise_and(cornerMap,localMax,cornerMap);
	return cornerMap;
}

void HarrisDetector::getCorners(vector<cv::Point>& points,double qualityLevel)
{
	cv::Mat cornerMap = getCornerMap(qualityLevel);
	getCorners(points,cornerMap);
}

void HarrisDetector::getCorners(vector<cv::Point>& points,const cv::Mat& cornerMap)
{
	for(int y = 0;y < cornerMap.rows;y++)
	{
		const uchar *  rowPtr = cornerMap.ptr<uchar>(y);
		for(int x = 0;x < cornerMap.cols;x++)
		{
			if(rowPtr[x])
			{
				points.push_back(cv::Point(x,y));
			}
		}
	}
}

void HarrisDetector::drawOnImage(cv::Mat &image,const vector<cv::Point>& points,cv::Scalar color,int radius,int thickness)
{
	vector<cv::Point>::const_iterator it = points.begin();
	while(it != points.end())
	{
		cv::circle(image,*it,radius,color,thickness);
		++it;
	}
}

void HarrisDetector::getCorners(vector<cv::KeyPoint>& points,double qualityLevel)
{
	cv::Mat cornerMap = getCornerMap(qualityLevel);
	getCorners(points,cornerMap);
}

void HarrisDetector::getCorners(vector<cv::KeyPoint>& points,const cv::Mat& cornerMap)
{
	for(int y = 0;y < cornerMap.rows;y++)
	{
		const uchar *  rowPtr = cornerMap.ptr<uchar>(y);
		for(int x = 0;x < cornerMap.cols;x++)
		{
			if(rowPtr[x])
			{
				points.push_back(cv::KeyPoint(x,y,-1));
			}
		}
	}
}
