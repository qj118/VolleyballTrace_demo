#include "stdafx.h"
#include "HarrisDetector.h"

Rect KalmanPre(CvKalman* kalman,const Rect& box)
{
	const CvMat* prediction=cvKalmanPredict(kalman,0);
	Point predict_pt=Point((int)prediction->data.fl[0],(int)prediction->data.fl[1]);
	Rect trackWindow = Rect(predict_pt.x -box.width,predict_pt.y - box.height, 2*box.width, 2*box.height);
	return trackWindow;
}

int KalmanUpdate(CvKalman* kalman,CvMat* measurement)
{
	cvKalmanCorrect(kalman,measurement);
	return 0;
}