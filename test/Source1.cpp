#include <iostream>
#include <opencv2\opencv.hpp>
#include <stdlib.h>
#include <math.h>
using namespace std;
using namespace cv;



Point m_nonWP;
vector<Point> m_omega;
Mat m_image;
Mat m_filteredImage;
Mat_<Vec3f> m_imageInLab;
vector<Vec3f> FV;

void filtering(Mat & image, Mat & filteredImage) {

	bilateralFilter(image, filteredImage, 5, 150, 150);
}

void computeOmega(int radius) {
	for (int i = 0; i < m_image.rows; i++) {
		for (int j = 0; j < m_image.cols; j++) {
			Point courant(i, j);
			if (norm(courant - m_nonWP) < radius) m_omega.push_back(courant);
		}
	}
}

void computeFV() {

		Mat_<Vec3f> imageLocalFloat;
		m_image.convertTo(imageLocalFloat, CV_32FC3, 1.0f / 255.0f);
		cvtColor(imageLocalFloat, m_imageInLab, CV_BGR2Lab);


		filtering(m_imageInLab, m_filteredImage);

		namedWindow("Image fiiltrée", WINDOW_AUTOSIZE);
		imshow("Image filtrée", m_filteredImage);

		Mat lab[3];
		split(m_filteredImage, lab);

		for (auto courant = m_omega.begin(); courant != m_omega.end(); courant++)
		{
			Vec3f candidat(lab[0].at<float>(*courant), lab[1].at<float>(*courant), lab[2].at<float>(*courant));
			FV.push_back(candidat);
		}
}

void computeDP() {

		float maxValue = 0;
		Mat dp(m_image.rows, m_image.cols, CV_32FC1);
	
		Mat lab[3];
		split(m_imageInLab, lab);
		for (int i = 0; i < m_image.rows; i++) {
			for (int j = 0; j < m_image.cols; j++) {
	
				Vec3f courant(lab[0].at<float>(i, j), lab[1].at<float>(i, j), lab[2].at<float>(i, j));
	
				float value = 0;
				float resultnorm = 0;
				for (int k = 0; k < FV.size(); k++) {
					resultnorm += norm(courant - FV[k]);


				}


				dp.at<float>(i, j) = resultnorm;
				//cout << value << endl;
				if (value >= maxValue) {
					maxValue = value;
				}
			
	
			}
	
			
		}
		cout << " voila la maxvalue  : " << maxValue << endl;
		Mat todisplay;// = dp / maxValue;
	//	cvtColor(dp, imageInLab, CV_BGR2Lab);
		cv::normalize(dp, todisplay, 0, 255, NORM_MINMAX,CV_8U);
	//	todisplay = norm_0_255(dp);
		namedWindow("hop", WINDOW_AUTOSIZE);// Create a window for display.
		imshow("hop",todisplay);
}

void on_mouse(int e, int x, int y, int d, void *ptr)
{
	if (e == EVENT_LBUTTONDOWN)
	{
		m_nonWP = Point(x, y);
		cout << "Le point selectionné a pour coordonnées : " << m_nonWP.x << " " << m_nonWP.y << endl;
		cv::setMouseCallback("Result", NULL, NULL);
		cout << "Lancement des calculs" << endl;
		computeOmega(10);
		computeFV();
		computeDP();

	}
}






int main() {


	char* imageName;
	imageName = "../test1.jpg";
	m_image = imread(imageName, 1);

	
	namedWindow("Image Depart", WINDOW_AUTOSIZE);
	imshow("Image Depart", m_image);



	setMouseCallback("Image Depart", on_mouse, NULL);
	
	while (1)
	{
		int key = cvWaitKey(10);
		if (key == 27) break;
	}
	return 0;


}