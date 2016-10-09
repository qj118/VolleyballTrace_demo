// Match.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"
#include "HarrisDetector.h"
#include <opencv2\features2d\features2d.hpp>
#include <opencv2\legacy\legacy.hpp>
#include <cstdio>


Point Match(const Mat& descriptor,const vector<cv::KeyPoint>& pts,const CvRect& box,const CvRect& trackWindow,IplImage* frame)
{
	HarrisDetector harris;
	Mat imageGray,image;
	vector<cv::KeyPoint> pts_2;
	CvRect matchBox = Rect(trackWindow.x,trackWindow.y,box.width,box.height);
	cv::BruteForceMatcher<cv::L2<float> >matcher;
	vector<cv::DMatch> matches;
	while(true)
	{
		bool matchFlag = false;
		cvSetImageROI(frame,matchBox);
	    image = frame;
	    cv::cvtColor(image,imageGray,CV_BGR2GRAY);
	    harris.detect(imageGray);
	    cv::Mat descriptor_2 = harris.getHarris();
	    harris.getCorners(pts_2,0.01);

	    matcher.match(descriptor,descriptor_2,matches);
		printf("Matches number: %d\n",matches.size());
		if(matches.size() >= 500)
		{
			matchFlag = true;
			cvResetImageROI(frame);
			break;
		}
		cvResetImageROI(frame);
		if(matchBox.width + box.width/2 <= trackWindow.width)
		{
			matchBox = Rect(matchBox.x + box.width/2,matchBox.y,box.width,box.height);
		}
		else if(matchBox.height + box.height/2 <= trackWindow.height)
		{
			matchBox = Rect(trackWindow.x,matchBox.y + box.height/2,box.width,box.height);
		}
		else
		{
			break;
		}
	}
	Point centerPoint = Point(matchBox.x + matchBox.width/2,matchBox.y + matchBox.height/2);
	return centerPoint;
}
