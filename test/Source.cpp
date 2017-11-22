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
Mat imageLab;
vector<Point> omega;
vector<f> fi;


void getOmega(int radius) {


	cout << " scribble 1 . x " << scribbles[0].x << endl;

	for (int i = 0; i < image.rows; i++) {
		for (int j = 0; j < image.cols; j++) {

			Point courant(i, j);
			if (norm(courant - scribbles[0]) < radius) {
				omega.push_back(courant);
				//img.at<uchar>(courant) = 0;
			}
			if (norm(courant - scribbles[1]) < radius) {
				omega.push_back(courant);
				//img.at<uchar>(courant) = 0;
			}
		}
	}

	cout << " taille de omega " << omega.size() << endl;


}


void computefi(float siga, float sigs) {

	cvtColor(image, imageLab, CV_BGR2Lab);

	Mat lab[3];
	split(imageLab, lab);
	int c = 0;
	cout << "omega" << omega.size() << endl;



	for (auto courant = omega.begin(); courant != omega.end(); courant++)
	{
		c++;
		f fcourant;
		fcourant.lisiga = lab[0].at<uchar>(*courant) / siga;
		fcourant.asiga = lab[1].at<uchar>(*courant) / siga;
		fcourant.bsiga = lab[2].at<uchar>(*courant) / siga;
		fcourant.xisigs = (*courant).x / sigs;
		fcourant.yisigs = (*courant).y / sigs;
		fi.push_back(fcourant);
	}

	cout << "c = " << c << " taille de fi " << fi.size() << endl;

}

void computeDP(float siga, float sigs) {


	Mat dp(image.cols, image.rows, CV_32F);

	Mat lab[3];
	split(imageLab, lab);
	for (int i = 0; i < image.rows; i++) {
		for (int j = 0; j < image.cols; j++) {

			if (lab[0].at<uchar>(i,j) != 0) {

			
			f fcourant;
			fcourant.lisiga = lab[0].at<uchar>(i, j) / siga;
			fcourant.asiga = lab[1].at<uchar>(i, j) / siga;
			fcourant.bsiga = lab[2].at<uchar>(i, j) / siga;
			fcourant.xisigs = i / sigs;
			fcourant.yisigs = j / sigs;

			float value = 0;

			for (int k = 0; k < fi.size(); k++) {
				value += sqrt(pow(fcourant.lisiga - fi[k].lisiga, 2)
					+ pow(fcourant.asiga - fi[k].asiga, 2)
					+ pow(fcourant.bsiga - fi[k].bsiga, 2)
					+ pow(fcourant.xisigs - fi[k].xisigs, 2)
					+ pow(fcourant.yisigs - fi[k].yisigs, 2));
				cout << "i = " << i << " j = " << j << " k  = " << k << endl;
			}

			dp.at<float>(i, j) = value;
		}
		}
	}

	namedWindow("hop", WINDOW_AUTOSIZE);// Create a window for display.
	imshow("hop", dp);
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
	imageName = "../projetCAVtest1.jpg";
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

