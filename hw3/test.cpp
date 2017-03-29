//--------------------------------------------------------------------------------------------------
// Linear time Maximally Stable Extremal Regions implementation as described in D. Nistér and H.
// Stewénius. Linear Time Maximally Stable Extremal Regions. Proceedings of the European Conference
// on Computer Vision (ECCV), 2008.
// 
// Copyright (c) 2012 Idiap Research Institute, http://www.idiap.ch/.
// Written by Charles Dubout <charles.dubout@idiap.ch>.
// 
// MSER is free software: you can redistribute it and/or modify it under the terms of the GNU
// General Public License version 3 as published by the Free Software Foundation.
// 
// MSER is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
// the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General
// Public License for more details.
// 
// You should have received a copy of the GNU General Public License along with MSER. If not, see
// <http://www.gnu.org/licenses/>.
//--------------------------------------------------------------------------------------------------
#define _USE_MATH_DEFINES 
#include <algorithm>

#include <cmath>
#include <ctime>
#include <cstdio>
#include <iostream>


#include <jpeglib.h>

#include "mser.h"

using namespace std;

#include <cv.h>
#include <highgui.h>
using namespace cv;


void drawEllipse(const MSER_1::Region & region, int width, int height, int depth,
				 vector<uint8_t> & bits, const uint8_t * color)
{
	// Centroid (mean)
	const double x = region.moments_[0] / region.area_;
	const double y = region.moments_[1] / region.area_;
	
	// Covariance matrix [a b; b c]
	const double a = region.moments_[2] / region.area_ - x * x;
	const double b = region.moments_[3] / region.area_ - x * y;
	const double c = region.moments_[4] / region.area_ - y * y;
	
	// Eigenvalues of the covariance matrix
	const double d  = a + c;
	const double e  = a - c;
	const double f  = sqrt(4.0 * b * b + e * e);
	const double e0 = (d + f) / 2.0; // First eigenvalue
	const double e1 = (d - f) / 2.0; // Second eigenvalue
	
	// Desired norm of the eigenvectors
	const double e0sq = sqrt(e0);
	const double e1sq = sqrt(e1);
	
	// Eigenvectors
	double v0x = e0sq;
	double v0y = 0.0;
	double v1x = 0.0;
	double v1y = e1sq;
	
	if (b) {
		v0x = e0 - c;
		v0y = b;
		v1x = e1 - c;
		v1y = b;
		
		// Normalize the eigenvectors
		const double n0 = e0sq / sqrt(v0x * v0x + v0y * v0y);
		v0x *= n0;
		v0y *= n0;
		
		const double n1 = e1sq / sqrt(v1x * v1x + v1y * v1y);
		v1x *= n1;
		v1y *= n1;
	}
	
	for (double t = 0.0; t < 2.0 * M_PI; t += 0.001) {
		int x2 = x + (cos(t) * v0x + sin(t) * v1x) * 2.0 + 0.5;
		int y2 = y + (cos(t) * v0y + sin(t) * v1y) * 2.0 + 0.5;
		
		if ((x2 >= 0) && (x2 < width) && (y2 >= 0) && (y2 < height))
			for (int i = 0; i < std::min(depth, 3); ++i)
				bits[(y2 * width + x2) * depth + i] = color[i];
	}
}


int main(void)
{
	// Try to load the jpeg image
	int width;
	int height;
	int depth;
	vector<uint8_t> original;
	//=========================
	//     ˇˇˇˇˇˇˇ
	//=========================
	//讀圖
	Mat image_old = imread( "sampleImage.jpg ");
	//imshow("image_old",image_old);
	//cvWaitKey(1);
	//寬,高,深度
	width = image_old.cols;
	cout <<  "width=" << width << endl;
	height = image_old.rows;
	cout <<  "height="<< height<< endl;
	depth = image_old.channels();
	cout <<  "depth="<< depth<< endl;
	
	//測試canny
	Mat dst1;
	clock_t start_time = clock();
	Canny(image_old, dst1, 50, 150, 3);
	clock_t end_time = clock();
	cout << "canny time" << (float)(end_time - start_time)/CLOCKS_PER_SEC << endl;

	/*printf("Time : %f sec \n", total_time);*/
	//imshow("Canny_1", dst1);
	//waitKey(1);

	for(int j = 0; j < height; ++j){
		for(int i = 0; i < width; ++i){
			for(int k = 0; k < depth; ++k){
				original.push_back(image_old.at<Vec3b>(j,i)[k]);
			}
		}
	}
	//=========================
	//     ^^^^^^^^^^^^^^^
	//=========================
	// Create a grayscale image
	vector<uint8_t> grayscale(width * height);
	
	for (int i = 0; i < width * height; ++i) {
		int sum = 0;
		
		for (int j = 0; j < depth; ++j){
			sum += static_cast<int>(original[i * depth + j]);
		}
		grayscale[i] = sum / depth;
	}
	//show出灰階圖驗證
	Mat gray_image = Mat::zeros(image_old.rows, image_old.cols, CV_8U);
	for (int gray_line=0; gray_line<image_old.rows;gray_line++){
		for (int gray_pixel=0; gray_pixel<image_old.cols;gray_pixel++){
			gray_image.at<uchar>(gray_line,gray_pixel) = grayscale[gray_line*image_old.cols + gray_pixel];
		}
	}
	//imshow("gray_image",gray_image);
	//cvWaitKey(1);

	// Extract MSER
	clock_t start = clock();

	/*MSER_1 mser8(2, 0.0005, 0.1, 0.5, 0.5, true);//原始CODE設定
	MSER_1 mser4(2, 0.0005, 0.1, 0.5, 0.5, false);*/
	MSER_1 mser8(5, 0.0001, 0.1, 0.25, 0.2, true);//網路上的默認值
	MSER_1 mser4(5, 0.0001, 0.1, 0.25, 0.2, false);

	
	vector<MSER_1::Region> regions[2];
	
	mser8(&grayscale[0], width, height, regions[0]);
	
	// Invert the pixel values
	for (int i = 0; i < width * height; ++i)
		grayscale[i] = ~grayscale[i];
	
	mser4(&grayscale[0], width, height, regions[1]);
	
	clock_t stop = clock();
	
	cout << "Extracted " << (regions[0].size() + regions[1].size()) << " regions from " << "argv[1]"
		 << " (" << width << 'x' << height << ") in "
		 << (static_cast<double>(stop - start) / CLOCKS_PER_SEC) << "s." << endl;
	
	 //Draw ellipses in the original image
	const uint8_t colors[2][3] = {{255, 0, 0}, {0, 0, 255}};

	Mat result_image = Mat::zeros(image_old.rows, image_old.cols, CV_8UC3);	
	
	for (int i = 0; i < 2; ++i){
		for (int j = 0; j < regions[i].size(); ++j){
			drawEllipse(regions[i][j], width, height, depth, original, colors[i]);
			
		}

	}
	for (int result_line=0; result_line<image_old.rows;result_line++){
		for (int result_pixel=0; result_pixel<image_old.cols;result_pixel++){
			for (int result_depth=0; result_depth<3;result_depth++){
			result_image.at<Vec3b>(result_line,result_pixel)[result_depth] = original[(result_line*image_old.cols + result_pixel)*3 + result_depth];
			}
		}
	}
	imshow("result_image",result_image);
	waitKey(1);

	system("PAUSE");
	destroyAllWindows();
	return 0;
}
