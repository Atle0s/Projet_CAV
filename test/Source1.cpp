#include <iostream>
#include <opencv2\opencv.hpp>
#include <stdlib.h>
#include <math.h>
#include <numeric>
using namespace std;
using namespace cv;



Point m_nonWP;
vector<Point> m_omega;
Mat m_image;
Mat m_filteredImage;
Mat_<Vec3f> m_imageInLab;
vector<Vec3f> FV;

void filtering(Mat & image, Mat & filteredImage) {

	Mat tmp;
	image.copyTo(tmp);
	for (int i = 0; i < 5; i++) {
		bilateralFilter(tmp, filteredImage, -1, 2, 2);
		filteredImage.copyTo(tmp);

		//cvtColor(imageLocalFloat, m_imageInLab, CV_BGR2Lab);
	}
	
}

vector<Point> tirage(const Mat& picture, vector<float>& prob_x, vector< vector<float> >& prob_y, const int& height, const int& width, int numberPoint) 
{

	Mat src;

	src = picture.clone();

	int count = 0;
	vector<int> count_y(prob_x.size());

	// We count pixels on the edges
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{

				prob_x[i] = prob_x[i] + 1;
				prob_y[i][j] = 1;
				count = count + 1;
				count_y[i] = count_y[i] + 1;
		
		}
	}

	// We divided by the count to create a probability
	std::transform(prob_x.begin(), prob_x.end(), prob_x.begin(), std::bind2nd(std::divides<float>(), count));

	for (unsigned int i = 0; i < count_y.size(); i++)
	{
		if (count_y[i] != 0)
		{
			std::transform(prob_y[i].begin(), prob_y[i].end(), prob_y[i].begin(), std::bind2nd(std::divides<float>(), count_y[i]));
		}
	}

	// Cumulative sum
	partial_sum(prob_x.begin(), prob_x.end(), prob_x.begin(), plus<double>());
	for (unsigned int i = 0; i < count_y.size(); i++)
	{
		partial_sum(prob_y[i].begin(), prob_y[i].end(), prob_y[i].begin(), plus<double>());
	}

	vector<Point> result;
	for (int i = 0; i < numberPoint; i++){

		double rand_x = rand() / (double)RAND_MAX;
		double rand_y = rand() / (double)RAND_MAX;

		int x = (upper_bound(prob_x.begin(), prob_x.end(), rand_x) - prob_x.begin());
		int y = (upper_bound(prob_y[x].begin(), prob_y[x].end(), rand_y) - prob_y[x].begin());
		result.push_back(Point(y, x));
	}
	return result;
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

		//namedWindow("Image fiiltr�e", WINDOW_AUTOSIZE);
		

		Mat lab[3];
		split(m_filteredImage, lab);
	
	//	imshow("Image filtr�e", m_imageInLab);
		for (auto courant = m_omega.begin(); courant != m_omega.end(); courant++)
		{
			Vec3f candidat(lab[0].at<float>(*courant), lab[1].at<float>(*courant), lab[2].at<float>(*courant));
			FV.push_back(candidat);
		}
}


//return DP mat (normalize et CV_8U)
Mat computeDP() {

		float maxValue = 0;
		Mat dp(m_image.rows, m_image.cols, CV_32FC1);
	
		Mat lab[3];
		split(m_filteredImage, lab);
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
		imshow("hop", todisplay);
		return todisplay;
}

void findPointsToWheather(Mat & wheatheringMap, int nombre) {
	vector<float> prob_x(wheatheringMap.rows);
	vector<vector<float>> prob_y(wheatheringMap.rows, vector<float>(wheatheringMap.cols));
	vector<Point> pointsToWheather = tirage(wheatheringMap, prob_x, prob_y, wheatheringMap.rows, wheatheringMap.cols, 15);
	/*
	Mat todisplay;
	wheatheringMap.convertTo(todisplay, CV_8UC3);
	for (int i = 0; i < pointsToWheather.size(); i++) {
		cout << pointsToWheather[i].x << " " << pointsToWheather[i].y << endl;
		todisplay.at<Vec3i>(pointsToWheather[i].x, pointsToWheather[i].y)[0] = 255;
	}

	namedWindow("hophop", WINDOW_AUTOSIZE);// Create a window for display.
	imshow("hophop", todisplay);
	*/
}

void weatherPoints(const std::vector<cv::Point> pointsToWeather, const cv::Mat& patch)
{
	cv::Mat map;
	cv::Mat weatheredImage(m_image.rows, m_image.cols, CV_32FC3);
	for (int p = 0; p < pointsToWeather.size(); p++)
	{
		cv::Point currentPoint = pointsToWeather[p];
		for (int i_patch = std::floor(currentPoint.x - patch.rows / 2); i_patch < std::floor(currentPoint.x + patch.rows / 2); i_patch++)
		{
			for (int j_patch = std::floor(currentPoint.y - patch.cols / 2); j_patch < std::floor(currentPoint.y + patch.cols / 2); j_patch++)
			{
				weatheredImage.at<Vec3f>(i_patch, j_patch)[0] = (1 - map.at<float>(currentPoint)) * m_image.at<Vec3f>(i_patch, j_patch)
					+ map.at<float>(currentPoint) * patch.at<Vec3f>(i_patch, j_patch);
			}
		}
	}
}

void on_mouse(int e, int x, int y, int d, void *ptr)
{
	if (e == EVENT_LBUTTONDOWN)
	{
		m_nonWP = Point(x, y);
		cout << "Le point selectionn� a pour coordonn�es : " << m_nonWP.x << " " << m_nonWP.y << endl;
		cv::setMouseCallback("Result", NULL, NULL);
		cout << "Lancement des calculs" << endl;
		computeOmega(10);
		computeFV();
		Mat dp = computeDP();
		findPointsToWheather(dp, 15);
		
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