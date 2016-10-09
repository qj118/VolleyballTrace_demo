#include "stdafx.h"
#include "CameraCalib.h"
#include <ctime>
#include <cstring>

CameraCalib::CameraCalib()
{
}

CameraCalib::~CameraCalib()
{
}

int CameraCalib::initCornerData(int nImages,cv::Size imageSize,cv::Size boardSize,float squareWidth,CornerDatas& cornerDatas)
{
	cornerDatas.nImages = nImages;
	cornerDatas.imageSize = imageSize;
	cornerDatas.boardSize = boardSize;
	cornerDatas.nPoints = nImages * boardSize.width * boardSize.height;
	cornerDatas.nPointsPerImage = boardSize.width * boardSize.height;
	cornerDatas.objectPoints.resize(nImages, vector<cv::Point3f>(cornerDatas.nPointsPerImage, cv::Point3f(0,0,0)));
	cornerDatas.imagePoints.resize(nImages, vector<cv::Point2f>(cornerDatas.nPointsPerImage, cv::Point2f(0,0)));
	
	int i,j,k,n;
	for(i = 0;i < nImages;i++)
	{
		n = 0;
		for(j = 0;j < boardSize.height;j++)
		{
			for(k = 0;k < boardSize.width;k++)
			{
				cornerDatas.objectPoints[i][n++] = cv::Point3f(j*squareWidth,k*squareWidth,0);
			}
		}
	}

	return 1;
}

int CameraCalib::resizeCornerData(int nImages,CornerDatas& cornerDatas)
{
	cornerDatas.nImages = nImages;
	cornerDatas.nPoints = nImages * cornerDatas.nPointsPerImage;
	cornerDatas.objectPoints.resize(nImages);
	cornerDatas.imagePoints.resize(nImages);
	return 1;
}

int CameraCalib::loadCornerData(const char* filename,CornerDatas& cornerDatas)
{
	cv::FileStorage fs(filename,cv::FileStorage::READ);
	if(fs.isOpened())
	{
		fs["nPoints"] >> cornerDatas.nPoints;
		fs["nImages"] >> cornerDatas.nImages;
		fs["nPointsPerImage"] >> cornerDatas.nPointsPerImage;

		cv::FileNodeIterator it = fs["imageSize"].begin();
		it >> cornerDatas.imageSize.width >> cornerDatas.imageSize.height;

		cv::FileNodeIterator bt = fs["boardSize"].begin();
		it >> cornerDatas.boardSize.width >> cornerDatas.boardSize.height;

		for(int i = 0;i < cornerDatas.nImages;i++)
		{
			stringstream imagename;
			imagename << "image" << i;

			cv::FileNode img = fs[imagename.str()];
			vector<cv::Point3f> ov;
			vector<cv::Point2f> iv;
			for(int j = 0;j < cornerDatas.nPointsPerImage;j++)
			{
				stringstream nodename;
				nodename << "node" << j;
				
				cv::FileNode pnt = img[nodename.str()];
				cv::Point3f op;
				cv::Point2f ip;
				cv::FileNodeIterator ot = pnt["objectPoints"].begin();
				ot >> op.x >> op.y >> op.z;
				cv::FileNodeIterator it =pnt["imagePoints"].begin();
				it >> ip.x >> ip.y;

				iv.push_back(ip);
				ov.push_back(op);
			}
			cornerDatas.imagePoints.push_back(iv);
			cornerDatas.objectPoints.push_back(ov);
		}
		fs.release();
		return 1;
	}
	else
	{
		return 0;
	}
}

int CameraCalib::saveCornerData(const char* filename,const CornerDatas& cornerDatas)
{
	cv::FileStorage fs(filename,cv::FileStorage::WRITE);
	if(fs.isOpened())
	{
		time_t rawtime;
		time(&rawtime);
		fs << "calibrationDate" << asctime(localtime(&rawtime));
		fs <<"nPoints" << cornerDatas.nPoints;
		fs << "nImages" << cornerDatas.nImages;
		fs << "nPointsPerImage" << cornerDatas.nPointsPerImage;
		fs << "imageSize" << "[" << cornerDatas.imageSize.width << cornerDatas.imageSize.height << "]";
		fs << "boardSize" << "[" << cornerDatas.boardSize.width << cornerDatas.boardSize.height << "]";

		for(int i = 0;i < cornerDatas.nImages;i++)
		{
			stringstream imagename;
			imagename << "image" << i;
			fs  << imagename.str() << "{";
			
			for(int j = 0;j < cornerDatas.nPointsPerImage;j++)
			{
				stringstream nodename;
				nodename << "node" << j;
				fs << nodename.str() << "{";
				cv::Point3f op = cornerDatas.objectPoints[i][j];
				cv::Point2f ip = cornerDatas.imagePoints[i][j];

				fs << "objectPoints" << "[:";
				fs << op.x << op.y << op.z << "]";
				fs << "imagePoints" << "[:";
				fs << ip.x << ip.y << "]";
				fs << "}";
			}
			fs << "}";
		}
		fs.release();
		return 1;
	}
	else{
		return 0;
	}
}

int CameraCalib::detectCorners(cv::Mat& img,CornerDatas& cornerDatas,int imageCount)
{
	vector<cv::Point2f>& corners = cornerDatas.imagePoints[imageCount];
	bool found = false;
	int flags = cv::CALIB_CB_ADAPTIVE_THRESH + cv::CALIB_CB_NORMALIZE_IMAGE + cv::CALIB_CB_FAST_CHECK;
	found = findChessboardCorners(img,cornerDatas.boardSize,corners,flags);
	if(found)
	{
		//cv::Mat gray;
		//cvtColor(img,gray,COLOR_RGB2GRAY);
		cv::Size regionSize(11,11);
		cornerSubPix(img,corners,regionSize,cv::Size(-1,-1),cv::TermCriteria(cv::TermCriteria::EPS | cv::TermCriteria::MAX_ITER,30,0.05));
	}
	drawChessboardCorners(img,cornerDatas.boardSize,corners,found);
	char info[10];
	sprintf_s(info,"%02d/%02d",imageCount + 1,cornerDatas.nImages);
	string text = info;
	showText(img,text);
	return found ? 1 : 0;
}

void CameraCalib::showText(cv::Mat& img,string text)
{
	int fontFace = cv::FONT_HERSHEY_PLAIN;
	double fontScale = 1;
	int fontThickness = 2;

	int textBaseline = 0;
	cv::Size textSize = cv::getTextSize(text,fontFace,fontScale,fontThickness,&textBaseline);
	textBaseline += fontThickness;

	cv::Point textOrg((img.cols - textSize.width - 10),(img.rows - textSize.height - 10));

	rectangle(img,textOrg + cv::Point(0,textBaseline),textOrg + cv::Point(textSize.width,-textSize.height),cv::Scalar(0,255,0));

	line(img,textOrg + cv::Point(0,fontThickness),textOrg + cv::Point(textSize.width,fontThickness),cv::Scalar(0,255,0));

	putText(img,text,textOrg,fontFace,fontScale,cv::Scalar(0,0,255),fontThickness,8);
}

int CameraCalib::loadCameraParams(const char* filename,CameraParams& cameraParams)
{
	cv::FileStorage fs(filename,cv::FileStorage::READ);
	if(fs.isOpened())
	{
		cv::FileNodeIterator it = fs["imageSize"].begin();
		it >> cameraParams.imageSize.width >> cameraParams.imageSize.height;

		fs["cameraMatrix"] >> cameraParams.cameraMatrix;
		fs["distortionCoefficients"]	>> cameraParams.distortionCoefficients;
		fs["flags"]						>> cameraParams.flags;

		int nImages = 0;
		fs["nImages"] >> nImages;

		for(int i = 0;i < nImages;i++)
		{
			char matName[50];
			sprintf_s(matName,"rotationMatrix_%d",i);

			cv::Mat rotMat;
			fs[matName] >> rotMat;
			cameraParams.rotations.push_back(rotMat);
		}

		for (int i = 0; i < nImages; i++)
		{
			char matName[50];
			sprintf_s(matName, "translationMatrix_%d", i);

			cv::Mat tranMat;
			fs[matName] >> tranMat;
			cameraParams.translations.push_back(tranMat);
		}

		fs.release();
		return 1;
	}
	else{
		return 0;
	}
}

int CameraCalib::saveCameraParams(const CameraParams& cameraParams,const char* filename/* = "cameraParams.yml"*/)
{
	string filename_ = filename;
	if(filename_ == "")
	{
		int strLen = 20;
		char *pCurrTime = (char*)malloc(sizeof(char)* strLen);
		memset(pCurrTime,0,sizeof(char) * strLen);
		time_t now;
		time(&now);
		strftime(pCurrTime,strLen,"%Y_%m_%d_%H_%M_%S_",localtime(&now));

		filename_ = pCurrTime;
		filename_ += "cameraParams.yml";
	}

	cv::FileStorage fs(filename_.c_str(),cv::FileStorage::WRITE);
	if(fs.isOpened())
	{
		time_t rawtime;
		time(&rawtime);
		fs << "calibrationDate" << asctime(localtime(&rawtime));
		char flagText[1024];
		sprintf_s( flagText, "flags: %s%s%s%s%s",
			cameraParams.flags & cv::CALIB_FIX_K3 ? "fix_k3" : "",
			cameraParams.flags & cv::CALIB_USE_INTRINSIC_GUESS ? " + use_intrinsic_guess" : "",
			cameraParams.flags & cv::CALIB_FIX_ASPECT_RATIO ? " + fix_aspect_ratio" : "",
			cameraParams.flags & cv::CALIB_FIX_PRINCIPAL_POINT ? " + fix_principal_point" : "",
			cameraParams.flags & cv::CALIB_ZERO_TANGENT_DIST ? " + zero_tangent_dist" : "" );
		cvWriteComment(*fs, flagText, 0);

		fs << "flags" << cameraParams.flags;

		fs << "imageSize" << "[" << cameraParams.imageSize.width << cameraParams.imageSize.height << "]";
		fs << "cameraMatrix"			<< cameraParams.cameraMatrix;
		fs << "distortionCoefficients"	<< cameraParams.distortionCoefficients;

		int nImages = cameraParams.rotations.size();
		fs << "nImage" << nImages;
		for(int i = 0;i < nImages;i++)
		{
			char matName[50];
			sprintf_s(matName, "rotationMatrix_%d", i);

			fs << matName << cameraParams.rotations[i];
		}
		for (int i = 0; i < nImages; i++)
		{
			char matName[50];
			sprintf_s(matName, "translationMatrix_%d", i);

			fs << matName << cameraParams.translations[i];
		}
		fs.release();
		return 1;
	}
	else
	{
		return 0;
	}
}

int CameraCalib::calibrateSingleCamera(CornerDatas& cornerDatas, CameraParams& cameraParams)
{
	cameraParams.imageSize = cornerDatas.imageSize;

	cv::calibrateCamera(
		cornerDatas.objectPoints, 
		cornerDatas.imagePoints, 
		cornerDatas.imageSize, 
		cameraParams.cameraMatrix, 
		cameraParams.distortionCoefficients,
		cameraParams.rotations, 
		cameraParams.translations,
		cameraParams.flags
		);

	return 1;
}

int CameraCalib::getCameraCalibrateError(vector<vector<cv::Point3f> >& _objectPoints,vector<vector<cv::Point2f> >& _imagePoints,CameraParams& cameraParams,double& err)
{
	cv::Mat imagePoints2;
	int totalPoints = 0;
	double totalErr = 0;
	
	size_t nImages = _objectPoints.size();
	for(int i = 0;i < nImages;i++)
	{
		vector<cv::Point3f>& objectPoints = _objectPoints[i];
		vector<cv::Point2f>& imagePoints = _imagePoints[i];
		totalPoints += objectPoints.size();

		projectPoints(objectPoints,cameraParams.rotations[i],cameraParams.translations[i],cameraParams.cameraMatrix,cameraParams.distortionCoefficients,imagePoints2);

		cv::Mat imagePoint1 = cv::Mat(imagePoints);
		double erri = norm(imagePoints,imagePoints2,cv::NORM_L2);
		totalErr += erri * erri;
	}

	err = sqrt(totalErr/totalPoints);
	return 1;
}

int CameraCalib::rectifySingleCamera(CameraParams& cameraParams,RemapMatrixs& remapMatrixs)
{
cv:initUndistortRectifyMap(cameraParams.cameraMatrix,cameraParams.distortionCoefficients,cv::Mat(),
	   getOptimalNewCameraMatrix(cameraParams.cameraMatrix,cameraParams.distortionCoefficients,cameraParams.imageSize,1,cameraParams.imageSize,0),
	   cameraParams.imageSize,CV_16SC2,remapMatrixs.mX,remapMatrixs.mY);

   return 1;
}

int CameraCalib::remapImage(cv::Mat& img,cv::Mat& imgout,RemapMatrixs& remapMatrixs)
{
	if ( !remapMatrixs.mX.empty() && !remapMatrixs.mY.empty() )
	{
		cv::remap( img, imgout, remapMatrixs.mX, remapMatrixs.mY, cv::INTER_LINEAR );
		return 1;
	}
	return 0;
}

int Calib(CameraCalib& camera,CameraCalib::CornerDatas& cornerDatas,CameraCalib::CameraParams& cameraParams,CameraCalib::RemapMatrixs& remapMatrixs)
{
	    camera.setWorkDir("data/image/");
     	int nImages = 9;
		cv::Mat originImage = imread(camera.m_workDir + "000.jpg");
//		namedWindow("originImage",1);
//		imshow("originImage",originImage);
	    camera.initCornerData(nImages,cv::Size(originImage.cols,originImage.rows),cv::Size(6,8),2.7f,cornerDatas);
	    string filelist[9] = {"000.jpg","001.jpg","002.jpg","003.jpg","004.jpg","005.jpg","006.jpg","007.jpg","008.jpg"};
	    for(int i = 0;i < nImages;i++)
	    {
			cv::Mat image = imread(camera.m_workDir + filelist[i],0);
		   camera.detectCorners(image,cornerDatas,i);
	    }
		camera.saveCornerData("data/Calibration/CornerData.yml",cornerDatas);
        camera.calibrateSingleCamera(cornerDatas,cameraParams);
	    camera.saveCameraParams(cameraParams,"data/Calibration/CameraParams.yml");
	    camera.rectifySingleCamera(cameraParams,remapMatrixs);
	    cv::Mat remapImage;
	    camera.remapImage(originImage,remapImage,remapMatrixs);
 //    	namedWindow("remap",1);
//	    imshow("remap",remapImage);

	/*	camera.loadCornerData("CornerData.yml",cornerDatas);
	    camera.loadCameraParams("CameraParams.yml",cameraParams);
		camera.rectifySingleCamera(cameraParams,remapMatrixs);
		 cv::Mat remapImage1;
	    camera.remapImage(originImage,remapImage1,remapMatrixs);
		namedWindow("remap1",1);
	    imshow("remap1",remapImage1);*/
	return 0;
}

