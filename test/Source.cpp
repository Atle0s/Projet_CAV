#include <iostream>
#include <opencv2\opencv.hpp>
#include <stdlib.h>
#include <math.h>
using namespace std;
using namespace cv;


struct f
{
	float lisiga;
	float asiga;
	float bsiga;
	float xisigs;
	float yisigs;
};



vector<Point> scribbles;
int nbclic = 0;
Mat image;
//Mat imageLab;
Mat_<Vec3f> imageInLab;
vector<Point> omega;
vector<f> fi;
vector<float> testavecfloat;

//=======================================================================================
// static Mat norm_0_255(InputArray _src)
// Create and return normalized image
//=======================================================================================
static Mat norm_0_255(InputArray _src) {
	Mat src = _src.getMat();
	// Create and return normalized image:
	Mat dst;
	switch (src.channels()) {
	case 1:
		normalize(_src, dst, 0, 255, NORM_MINMAX, CV_8UC1);
		break;
	case 3:
		normalize(_src, dst, 0, 255, NORM_MINMAX, CV_8UC3);
		break;
	default:
		src.copyTo(dst);
		break;
	}
	return dst;
}


void getOmega(int radius) {


	cout << " scribble 1 . x " << scribbles[0].x << endl;

	for (int i = 0; i < image.rows; i++) {
		for (int j = 0; j < image.cols; j++) {

			Point courant(i, j);
			if (norm(courant - scribbles[0]) < radius) {
				omega.push_back(courant);
				
			}
			if (norm(courant - scribbles[1]) < radius) {
				omega.push_back(courant);
		
			}
		}
	}

	cout << "Taille de omega " << omega.size() << endl;


}


void computefi(float siga, float sigs) {


	Mat_<Vec3f> imageLocalFloat;
	image.convertTo(imageLocalFloat, CV_32FC3, 1.0f / 255.0f);

	// Conversion RGB to Lab

	cvtColor(imageLocalFloat, imageInLab, CV_BGR2Lab);



	//cvtColor(image, imageLab, CV_BGR2Lab);

	Mat lab[3];
	split(imageInLab, lab);
	int c = 0;
	cout << "Omega" << omega.size() << endl;



	for (auto courant = omega.begin(); courant != omega.end(); courant++)
	{
		c++;
		//testavecfloat.push_back(lab[0].at<float>(*courant));
		//cout << lab[0].at<float>(*courant) << endl;
		//cout << lab[2].at<float>(*courant) << endl;
		f fcourant;
		fcourant.lisiga = lab[0].at<float>(*courant) / siga;
		fcourant.asiga = lab[1].at<float>(*courant) / siga;
		fcourant.bsiga = lab[2].at<float>(*courant) / siga;
		fcourant.xisigs = (*courant).x / sigs;
		fcourant.yisigs = (*courant).y / sigs;
		fi.push_back(fcourant);
	}
	cout << "c = " << c << " taille de fi " << fi.size() << endl;

}

void computeDP(float siga, float sigs) {

	float maxValue = 0;
	Mat dp(image.rows, image.cols, CV_32FC1);

	Mat lab[3];
	split(imageInLab, lab);
	for (int i = 0; i < image.rows; i++) {
		for (int j = 0; j < image.cols; j++) {

			if (lab[0].at<float>(i,j) != 0.0f) {

			//	float courant = lab[0].at<float>(i, j) / siga;
			f fcourant;
			fcourant.lisiga = lab[0].at<float>(i, j) / siga;
			fcourant.asiga = lab[1].at<float>(i, j) / siga;
			fcourant.bsiga = lab[2].at<float>(i, j) / siga;
			fcourant.xisigs = i / sigs;
			fcourant.yisigs = j / sigs;

			float value = 0;

			for (int k = 0; k < fi.size(); k++) {
				//value += sqrt(pow(courant - testavecfloat[k], 2));
				//value = 10;
				//cout << courant - testavecfloat[k] << endl;

				//out << "Courant : " << lab[0].at<float>(i, j) << " Testavecflot[k] : " << testavecfloat[k] << endl;
				value += sqrt(pow(fcourant.lisiga - fi[k].lisiga, 2)
					+ pow(fcourant.asiga - fi[k].asiga, 2)
					+ pow(fcourant.bsiga - fi[k].bsiga, 2)
					+ pow(fcourant.xisigs - fi[k].xisigs, 2)
					+ pow(fcourant.yisigs - fi[k].yisigs, 2));
				//cout << "i = " << i << " j = " << j << " k  = " << k << endl;
			
			}
			
			dp.at<float>(i, j) = value;
			if (value >= maxValue) {
				maxValue = value;
			}
		}

		}

		
	}
	cout << " voila la maxvalue  : " << maxValue << endl;
	Mat todisplay = dp / maxValue;
//	cvtColor(dp, imageInLab, CV_BGR2Lab);
	//cv::normalize(dp, todisplay, 0, 255, NORM_MINMAX,CV_8U);
//	todisplay = norm_0_255(dp);
	namedWindow("hop", WINDOW_AUTOSIZE);// Create a window for display.
	imshow("hop",todisplay);
}

void on_mouse(int e, int x, int y, int d, void *ptr)
{
	if (e == EVENT_LBUTTONDOWN)
	{
		if (scribbles.size() < 2)
		{
			Point courant(x, y);
			cout << courant.x << " " << courant.y << endl;
			scribbles.push_back(courant);
	
		}
		else
		{
		
			// On désactive le callback
			cv::setMouseCallback("Result", NULL, NULL);
			getOmega(5);
			computefi(0.2f, 50.0f);
			computeDP(0.2f, 50.0f);
		}
	}
}



int main() {

	
	char* imageName;
	//imageName = "C:\Users\leo-d\Documents\Visual Studio 2017\Projet_CAV/projetCAVtest1.jpg";
	imageName = "../test1.jpg";
	image = imread(imageName, 1);

	namedWindow("Result", WINDOW_AUTOSIZE);// Create a window for display.
	imshow("Result", image);

	
	setMouseCallback("Result", on_mouse, NULL);

	while (1)
	{
		int key = cvWaitKey(10);
		if (key == 27) break;
	}

//	waitKey(0);

	return 0;


}

