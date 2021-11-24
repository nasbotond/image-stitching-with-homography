// ImageStitchingWithHomography.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <stdio.h>  
#include <opencv2/opencv.hpp>  
#include <vector>
#include "opencv2/core.hpp"
#ifdef HAVE_OPENCV_XFEATURES2D
#include "opencv2/highgui.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"

// #include "MatrixReaderWriter.h"

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;
using std::cout;
using std::endl;

#define SQRT2 1.41

struct normalizedData {
	Mat T2D;
	vector<Point2f> newPts2D;
};


struct pair<normalizedData, normalizedData> NormalizeData(vector<pair<Point2f, Point2f>> pts2D) {
	
	int ptsNum = pts2D.size();

	//calculate means (they will be the center of coordinate systems)
	float mean1x = 0.0, mean1y = 0.0, meanx_tick = 0.0, meany_tick = 0.0;
	for (int i = 0; i < ptsNum; i++) {
		mean1x += pts2D[i].first.x;
        mean1y += pts2D[i].first.y;
		
		meanx_tick += pts2D[i].second.x;
		meany_tick += pts2D[i].second.y;
	}
	
	mean1x /= ptsNum;
	mean1y /= ptsNum;
	
	meanx_tick /= ptsNum;
	meany_tick /= ptsNum;

	float spread1x = 0.0, spread1y = 0.0, spreadx_tick = 0.0, spready_tick = 0.0;

	for (int i = 0; i < ptsNum; i++) {
		spread1x += (pts2D[i].first.x - mean1x) * (pts2D[i].first.x - mean1x);
		spread1y += (pts2D[i].first.y - mean1y) * (pts2D[i].first.y - mean1y);
		
		spreadx_tick += (pts2D[i].second.x - meanx_tick) * (pts2D[i].second.x - meanx_tick);
		spready_tick += (pts2D[i].second.y - meany_tick) * (pts2D[i].second.y - meany_tick);
	}

	spread1x /= ptsNum;
	spread1y /= ptsNum;
	
	spreadx_tick /= ptsNum;
	spready_tick /= ptsNum;

	Mat offs1, offs_tick = Mat::eye(3, 3, CV_32F);
	Mat scale1, scale_tick = Mat::eye(3, 3, CV_32F);

	offs1.at<float>(0, 2) = -mean1x;
	offs1.at<float>(1, 2) = -mean1y;
	
	offs_tick.at<float>(0, 2) = -meanx_tick;
	offs_tick.at<float>(1, 2) = -meany_tick;

	scale1.at<float>(0, 0) = SQRT2 / sqrt(spread1x);
	scale1.at<float>(1, 1) = SQRT2 / sqrt(spread1y);
	
	scale_tick.at<float>(0, 0) = SQRT2 / sqrt(spreadx_tick);
	scale_tick.at<float>(1, 1) = SQRT2 / sqrt(spready_tick);

	struct normalizedData ret, ret_tick;
	ret.T2D = scale1 * offs1;
	ret_tick.T2D = scale_tick * offs_tick;

	for (int i = 0; i < ptsNum; i++) {
		Point2d p2D;
		Point2d p2D_tick;

		p2D.x = SQRT2 * (pts2D[i].first.x - mean1x) / sqrt(spread1x);
		p2D.y = SQRT2 * (pts2D[i].first.y - mean1y) / sqrt(spread1y);
		
		p2D_tick.x = SQRT2 * (pts2D[i].second.x - meanx_tick) / sqrt(spreadx_tick);
		p2D_tick.y = SQRT2 * (pts2D[i].second.y - meany_tick) / sqrt(spready_tick);
		
		ret.newPts2D.push_back(p2D);
		ret_tick.newPts2D.push_back(p2D_tick);
	}

	pair<normalizedData, normalizedData> result;
	result.first = ret;
	result.second = ret_tick;
	return result;

}

Mat calcHomography(vector<pair<Point2f, Point2f>> pointPairs) {
	pair<normalizedData, normalizedData> norm = NormalizeData(pointPairs);
	
    const int ptsNum = pointPairs.size();
    Mat A(2 * ptsNum, 9, CV_32F);
    for (int i = 0; i < ptsNum; i++) {
        float u1 = norm.first.newPts2D[i].x;
        float v1 = norm.first.newPts2D[i].y;

        float u2 = norm.second.newPts2D[i].x;
        float v2 = norm.second.newPts2D[i].y;

        A.at<float>(2 * i, 0) = u1;
        A.at<float>(2 * i, 1) = v1;
        A.at<float>(2 * i, 2) = 1.0f;
        A.at<float>(2 * i, 3) = 0.0f;
        A.at<float>(2 * i, 4) = 0.0f;
        A.at<float>(2 * i, 5) = 0.0f;
        A.at<float>(2 * i, 6) = -u2 * u1;
        A.at<float>(2 * i, 7) = -u2 * v1;
        A.at<float>(2 * i, 8) = -u2;

        A.at<float>(2 * i + 1, 0) = 0.0f;
        A.at<float>(2 * i + 1, 1) = 0.0f;
        A.at<float>(2 * i + 1, 2) = 0.0f;
        A.at<float>(2 * i + 1, 3) = u1;
        A.at<float>(2 * i + 1, 4) = v1;
        A.at<float>(2 * i + 1, 5) = 1.0f;
        A.at<float>(2 * i + 1, 6) = -v2 * u1;
        A.at<float>(2 * i + 1, 7) = -v2 * v1;
        A.at<float>(2 * i + 1, 8) = -v2;

    }

    Mat eVecs(9, 9, CV_32F), eVals(9, 9, CV_32F);
    cout << A << endl;
    eigen(A.t() * A, eVals, eVecs);

    cout << eVals << endl;
    cout << eVecs << endl;


    Mat H(3, 3, CV_32F);
    for (int i = 0; i < 9; i++) H.at<float>(i / 3, i % 3) = eVecs.at<float>(8, i);

    cout << H << endl;

    //Normalize:
    H = H * (1.0 / H.at<float>(2, 2));
    cout << H << endl;

    return H;
}

const char* keys =
"{ help h |                  | Print help message. }"
"{ input1 | box.png          | Path to input image 1. }"
"{ input2 | box_in_scene.png | Path to input image 2. }";

int main(int argc, char* argv[])
{
    CommandLineParser parser(argc, argv, keys);
    Mat img1 = imread("Dev1_Image_w960_h600_fn1000.jpg", IMREAD_GRAYSCALE);
    Mat img2 = imread("Dev2_Image_w960_h600_fn1000.jpg", IMREAD_GRAYSCALE);
    if (img1.empty() || img2.empty())
    {
        cout << "Could not open or find the image!\n" << endl;
        parser.printMessage();
        return -1;
    }
    //-- Step 1: Detect the keypoints using SURF Detector, compute the descriptors
    int minHessian = 400;
    Ptr<SURF> detector = SURF::create(minHessian);
    std::vector<KeyPoint> keypoints1, keypoints2;
    Mat descriptors1, descriptors2;
    detector->detectAndCompute(img1, noArray(), keypoints1, descriptors1);
    detector->detectAndCompute(img2, noArray(), keypoints2, descriptors2);
    //-- Step 2: Matching descriptor vectors with a FLANN based matcher
    // Since SURF is a floating-point descriptor NORM_L2 is used
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
    std::vector< std::vector<DMatch> > knn_matches;
    matcher->knnMatch(descriptors1, descriptors2, knn_matches, 2);
    //-- Filter matches using the Lowe's ratio test
    const float ratio_thresh = 0.7f;
    std::vector<DMatch> good_matches;
    for (size_t i = 0; i < knn_matches.size(); i++)
    {
        if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance)
        {
            good_matches.push_back(knn_matches[i][0]);
        }
    }
    //-- Draw matches
    Mat img_matches;
    drawMatches(img1, keypoints1, img2, keypoints2, good_matches, img_matches, Scalar::all(-1),
        Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    //-- Show detected matches
    imshow("Good Matches", img_matches);
    waitKey();
    return 0;
}
#else
int main()
{
    std::cout << "This tutorial code needs the xfeatures2d contrib module to be run." << std::endl;
    return 0;
}
#endif
