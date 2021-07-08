#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>



int main(){
	cv::Mat image;
	image = cv::imread("1.jpg" ,cv::IMREAD_COLOR);
  
	if(! image.data ) {
		std::cout <<  "Image not found or unable to open" << std::endl ;
		return -1;
	}
  
	cv::namedWindow( "TEST UGRP", cv::WINDOW_AUTOSIZE );
	cv::imshow( "TEST UGRP", image );
  
	cv::waitKey(0);
	cv::destroyWindow("TEST UGRP");


	image = cv::imread("2.jpg", cv::IMREAD_COLOR);
	cv::namedWindow("TEST UGRP2", cv::WINDOW_AUTOSIZE);
	cv::imshow("TEST UGRP2", image);
	
	cv::waitKey(0);
	return 0;
}
