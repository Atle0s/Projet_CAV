#include <iostream>
#include <opencv2\opencv.hpp>
#include <stdlib.h>
#include <math.h>
#include <numeric>
#include <time.h>
using namespace std;
using namespace cv;



Point m_nonWP;
vector<Point> m_omega;
Mat m_image;
Mat m_filteredImage;
Mat_<Vec3f> m_imageInLab;
vector<Vec3f> FV;
int nbclic;

void filtering(Mat & image, Mat & filteredImage) {

	Mat tmp;
	image.copyTo(tmp);
	for (int i = 0; i < 10; i++) {
		bilateralFilter(tmp, filteredImage, -1, 7, 7);
		filteredImage.copyTo(tmp);

		//cvtColor(imageLocalFloat, m_imageInLab, CV_BGR2Lab);
	}
	
}

vector<Point> tirage(const Mat& picture, vector<float>& prob_x, vector< vector<float> >& prob_y, const int& height, const int& width, int numberPoint) 
{
	srand(clock());

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
		int y = (upper_bound(prob_y[x ].begin(), prob_y[x ].end(), rand_y) - prob_y[x ].begin());
		result.push_back(Point(y,x));
	}
	return result;
}

// Compute omega permet de créer l'ensemble de l'univers sur lequel on va créer notre carte.

void computeOmega(int radius) {
	for (int i = 0; i < m_image.rows; i++) {
		for (int j = 0; j < m_image.cols; j++) {
			Point courant(i, j);
			if (norm(courant - m_nonWP) < radius) m_omega.push_back(courant); // On compare la distance par rapport a un point non abimé.
		}
	}
}


// Ne fait que foutre des vec 3f avec LAB.
void computeFV() {

		Mat_<Vec3f> imageLocalFloat;
		m_image.convertTo(imageLocalFloat, CV_32FC3, 1.0f / 255.0f);
		cvtColor(imageLocalFloat, m_imageInLab, CV_BGR2Lab);


		filtering(m_imageInLab, m_filteredImage);

		//namedWindow("Image fiiltrée", WINDOW_AUTOSIZE);
		

		Mat lab[3];
		split(m_filteredImage, lab);
	
	//	imshow("Image filtrée", m_imageInLab);
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


		//bitwise_not(dp, dpTmp);

		cv::normalize(dp, todisplay, 0, 255, NORM_MINMAX,CV_8U);


namedWindow("hop", WINDOW_AUTOSIZE);// Create a window for display.
imshow("hop", todisplay);
return dp;
}

vector<Point> findPointsToWheather(Mat & wheatheringMap, int nombre) {
	vector<float> prob_x(wheatheringMap.rows);
	vector<vector<float>> prob_y(wheatheringMap.rows, vector<float>(wheatheringMap.cols));
	vector<Point> pointsToWheather = tirage(wheatheringMap, prob_x, prob_y, wheatheringMap.rows, wheatheringMap.cols, nombre);
	return pointsToWheather;
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
// patch = patch à recopier.


/*
void weatherPoints(const std::vector<cv::Point> pointsToWeather, const cv::Mat& patch, const cv::Mat& weatheringMap)
{
	cv::Mat weatheredImage(m_image.rows, m_image.cols, CV_32FC3); // result en RGB
	cv::Mat image_float(m_image.rows, m_image.cols, CV_32FC3);
	cv::normalize(m_image, image_float, 0, 255, NORM_MINMAX, CV_32FC3);

	weatheredImage = image_float.clone();

	cv::Mat normalizedDP;
	cv::normalize(weatheringMap, normalizedDP, 0, 1, NORM_MINMAX, CV_32FC1);
	//m_image.convertTo(image_float, CV_32FC3);

	//m_image.convertTo(image_float, CV_32FC3);
	//image_float = image_float / 255;

	for (int p = 0; p < pointsToWeather.size(); p++)
	{

		cv::Point currentPoint = pointsToWeather[p];
		for (int i_patch = std::floor(currentPoint.x - patch.rows / 2); i_patch < std::floor(currentPoint.x + patch.rows / 2); i_patch++)
		{
			int new_i = 0;
			for (int j_patch = std::floor(currentPoint.y - patch.cols / 2); j_patch < std::floor(currentPoint.y + patch.cols / 2); j_patch++)
			{
				int new_j = 0;
				if (!(i_patch < 0 || i_patch >= m_image.rows || j_patch < 0 || j_patch >= m_image.cols))
				{
					weatheredImage.at<Vec3f>(i_patch, j_patch)[0] = (1 - normalizedDP.at<float>(currentPoint)) * image_float.at<Vec3f>(i_patch, j_patch)[0] + normalizedDP.at<float>(currentPoint) *patch.at<Vec3f>(new_i, new_j)[0];
					weatheredImage.at<Vec3f>(i_patch, j_patch)[1] = (1 - normalizedDP.at<float>(currentPoint)) * image_float.at<Vec3f>(i_patch, j_patch)[1] + normalizedDP.at<float>(currentPoint) *patch.at<Vec3f>(new_i, new_j)[1];
					weatheredImage.at<Vec3f>(i_patch, j_patch)[2] = (1 - normalizedDP.at<float>(currentPoint)) * image_float.at<Vec3f>(i_patch, j_patch)[2] + normalizedDP.at<float>(currentPoint) * patch.at<Vec3f>(new_i, new_j)[2];
				//	cout << (1 - normalizedDP.at<float>(currentPoint)) * image_float.at<Vec3f>(i_patch, j_patch)[0] + normalizedDP.at<float>(currentPoint) * patch.at<Vec3f>(new_i, new_j)[0] << endl;
				}
				new_j++;
			}
			new_i++;
		}
	}
	imshow("en float ", weatheredImage);
	Mat todisplay;
	//	weatheredImage.convertTo(todisplay, CV_8UC3);
	weatheredImage.convertTo(todisplay, CV_8UC3);
	//cv::normalize(weatheredImage, todisplay, 0, 255, NORM_MINMAX, CV_8UC3);
	imshow("Result ", todisplay);
}

*/


Mat updateCarteProba(Mat & carteProba, vector<Point> oldPointsToWeather) {

	for (int i = 0; i < oldPointsToWeather.size(); i++) {
		carteProba.at<float>(oldPointsToWeather[i].x, oldPointsToWeather[i].y) = 0.0f;
	}

	return carteProba;
}


void mainFunction(Mat & image_float,Mat & carteProba, const cv::Mat& patch, const cv::Mat& weatheringMap) {
	cv::Mat weatheredImage(m_image.rows, m_image.cols, CV_32FC3); // result en RGB
	

	weatheredImage = image_float.clone();
	cv::Mat normalizedDP;
	cv::normalize(weatheringMap, normalizedDP, 0, 1, NORM_MINMAX, CV_32FC1);

	vector<Point> pointsToWeather = findPointsToWheather(carteProba, 100);
	updateCarteProba(carteProba, pointsToWeather);

	for (int p = 0; p < pointsToWeather.size(); p++)
	{

		cv::Point currentPoint = pointsToWeather[p];
		for (int i_patch = std::floor(currentPoint.x - patch.rows / 2); i_patch < std::floor(currentPoint.x + patch.rows / 2); i_patch++)
		{
			int new_i = 0;
			for (int j_patch = std::floor(currentPoint.y - patch.cols / 2); j_patch < std::floor(currentPoint.y + patch.cols / 2); j_patch++)
			{
				int new_j = 0;
				if (!(i_patch < 0 || i_patch >= m_image.rows || j_patch < 0 || j_patch >= m_image.cols))
				{
					weatheredImage.at<Vec3f>(i_patch, j_patch)[0] = (1 - normalizedDP.at<float>(currentPoint)) * image_float.at<Vec3f>(i_patch, j_patch)[0] + normalizedDP.at<float>(currentPoint) *patch.at<Vec3f>(new_i, new_j)[0];
					weatheredImage.at<Vec3f>(i_patch, j_patch)[1] = (1 - normalizedDP.at<float>(currentPoint)) * image_float.at<Vec3f>(i_patch, j_patch)[1] + normalizedDP.at<float>(currentPoint) *patch.at<Vec3f>(new_i, new_j)[1];
					weatheredImage.at<Vec3f>(i_patch, j_patch)[2] = (1 - normalizedDP.at<float>(currentPoint)) * image_float.at<Vec3f>(i_patch, j_patch)[2] + normalizedDP.at<float>(currentPoint) * patch.at<Vec3f>(new_i, new_j)[2];
					//	cout << (1 - normalizedDP.at<float>(currentPoint)) * image_float.at<Vec3f>(i_patch, j_patch)[0] + normalizedDP.at<float>(currentPoint) * patch.at<Vec3f>(new_i, new_j)[0] << endl;
				}
				new_j++;
			}
			new_i++;
		}
	}


	Mat todisplay;
	//	weatheredImage.convertTo(todisplay, CV_8UC3);
	weatheredImage.convertTo(todisplay, CV_8UC3);
	//cv::normalize(weatheredImage, todisplay, 0, 255, NORM_MINMAX, CV_8UC3);
	imshow("Result ", todisplay);
	
}





void on_mouse(int e, int x, int y, int d, void *ptr)
{
	if (e == EVENT_LBUTTONDOWN)
	{
		nbclic++;
		if (nbclic == 1) {
			m_nonWP = Point(x, y);
			cout << "Le point selectionné a pour coordonnées : " << m_nonWP.x << " " << m_nonWP.y << endl;
			cout << "Veuillez selectionner le patch svp" << endl;
			
		}
		else if (nbclic == 2) {
			
			Point centerPatch(x, y);

			Rect region_of_interest = Rect(centerPatch.x, centerPatch.y, 30, 30);

			cv::setMouseCallback("Result", NULL, NULL);
			cout << "Lancement des calculs" << endl;

			cout << "Création de l'univers Omega" << endl;
			computeOmega(10);

			cout << "Calcul des features vector" << endl;
			computeFV();


			cout << "Calcul de DP" << endl;
			Mat dp = computeDP();

			
			Mat dpProba = 1.0f - dp;
			//	imshow("dpTMP",dp);
			// inverser dp pour trouver les points .

			cv::Mat image_float(m_image.rows, m_image.cols, CV_32FC3);
			cv::normalize(m_image, image_float, 0, 255, NORM_MINMAX, CV_32FC3);


			Mat patch = image_float(region_of_interest);

			mainFunction(image_float, dpProba, patch, dp);
		//vector<Point> pointsToWeather = findPointsToWheather(dpProba, 100);

		//	for (int i = 0; i < pointsToWeather.size(); i++){
		///	cout << "Le point " << i << " a pour valeur " << pointsToWeather[i].x << " " << pointsToWeather[i].y << endl; 
			//}
			//Mat patch = Mat::ones(30, 30, CV_32FC3);
		
		//	imshow("patch", patch);
		///	weatherPoints(pointsToWeather, patch, dp);
			
		}
	
		
				
	}
}







int main() {


	char* imageName;
	imageName = "../test1.jpg";
	m_image = imread(imageName, 1);

	
	namedWindow("Image Depart", WINDOW_AUTOSIZE);
	imshow("Image Depart", m_image);

	nbclic = 0;

	setMouseCallback("Image Depart", on_mouse, NULL);
	
	while (1)
	{
		int key = cvWaitKey(10);
		if (key == 27) break;
	}
	return 0;


}