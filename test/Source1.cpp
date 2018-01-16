#include <iostream>
#include <opencv2\opencv.hpp>
#include <stdlib.h>
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

vector<Point> tirage(const Mat& img, int points_number) 
{
	cv::Mat norm_img;
	cv::normalize(img, norm_img, 0, 255, NORM_MINMAX, CV_32FC1);

	// Find the sum of all values to compute rows probabilities
	// and find the sums of all values in each row to compute
	// column knowing row probabilities
	
	float total_values_sum = 0.0f;
	std::vector<float> row_values_sum(norm_img.rows);
	for (int i = 0; i < norm_img.rows; i++)
	{
		row_values_sum[i] = 0.0f;

		for (int j = 0; j < norm_img.cols; j++)
		{
			total_values_sum += norm_img.at<float>(i, j);
			row_values_sum[i] += norm_img.at<float>(i, j);
		}
	}

	// Compute each row probability and store the cumulated
	// probability and store it in an array for later use.
	//
	// /!\ WARNING :
	//			The array is one cell longer than needed and
	//			is padded with a zero for the first	value to
	//			make resaerch easier later. 
	
	std::vector<float> row_cumulated_probs(norm_img.rows + 1);
	row_cumulated_probs[0] = 0.0f;
	
	for (int i = 0; i < norm_img.rows; i++)
	{
		// Start filling the array at the second cell
		row_cumulated_probs[i + 1] = row_cumulated_probs[i] + (row_values_sum[i] / total_values_sum);
	}

	// Compute column knowing row probability for each row
	// and store the results in a two dimensionnal array.

	std::vector<std::vector<float>> col_cumulated_probs_knowing_row(norm_img.rows);
	
	for (int i = 0; i < norm_img.rows; i++)
	{
		// /!\ WARNING :
		//			Same as earlier, the array is one cell longer
		//			and is padded with a zero.

		col_cumulated_probs_knowing_row[i] = std::vector<float>(norm_img.cols + 1);
		col_cumulated_probs_knowing_row[i][0] = 0.0f;

		for (int j = 0; j < norm_img.cols; j++)
		{
			col_cumulated_probs_knowing_row[i][j + 1] = col_cumulated_probs_knowing_row[i][j] + (norm_img.at<float>(i, j) / row_values_sum[i]);
		}
	}

	std::vector<cv::Point> result;
	for (int p = 0; p < points_number; p++)
	{
		// Put current system clock as random generator seed
		srand(clock());

		// Generate a random number and find where it falls in the
		// rows probabilities

		float rand_row = rand() / static_cast<float>(RAND_MAX);
		int drawn_row_index = 0;

		for (int i = 0; i < img.rows; i++)
		{
			if (row_cumulated_probs[i] <= rand_row && rand_row <= row_cumulated_probs[i +1])
			{
				drawn_row_index = i;
			}
		}
		
		// Generate a random number and find where it falls in the
		// column probabilities knowing row, knowing the row we
		// found just before
		
		float rand_col = rand() / static_cast<float>(RAND_MAX);
		int drawn_col_index = 0;
		
		for (int j = 0; j < img.cols; j++)
		{
			if (col_cumulated_probs_knowing_row[drawn_row_index][j] <= rand_col && rand_col <= col_cumulated_probs_knowing_row[drawn_row_index][j + 1])
			{
				drawn_col_index = j;
			}
		}

		result.push_back(cv::Point(drawn_row_index, drawn_col_index));
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
	vector<Point> pointsToWheather = tirage(wheatheringMap, nombre);
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

	vector<Point> pointsToWeather = findPointsToWheather(carteProba, 10);
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