#include <iostream>
#include <opencv2\opencv.hpp>
#include <stdlib.h>
#include <time.h>

// Attributes for algorithm parameters 
static int MAX_PATCH_SIZE = 50;
static int MAX_OMEGA_RADIUS = 50;
static int MAX_ITERATIONS = 100;
static int MAX_SAMPLES = 100;

int m_patch_size = 30;
int m_omega_radius = 10;
int m_iterations = 50;
int m_samples = 10;


// Attributes for runtime execution
cv::Mat m_src_img;
cv::Mat m_src_img_display;
std::vector<cv::Point> m_selected_points;

/**
* \brief
* \param img The image to work on
* \param weathered_point The center of the circle
* \param radius The radius of the circle
* \return A list of the points which are inside the circle of the specified radius centered on weathered_point.
*/
std::vector<cv::Point> compute_omega(const cv::Mat& img, const cv::Point& weathered_point, const int radius) {
	std::vector<cv::Point> omega;
	for (int x = 0; x < img.cols; x++) {
		for (int y = 0; y < img.rows; y++) {
			cv::Point current_point(x, y);
			if (cv::norm(current_point - weathered_point) < radius) omega.push_back(current_point);
		}
	}
	return omega;
}

/**
* \brief Converts the input image to LAB and apply several bilateral filters
* \param img
* \return A filtered image in LAB color space
*/
cv::Mat preprocess(const cv::Mat& img) {
	cv::Mat_<cv::Vec3f> temp_lab_img;
	img.convertTo(temp_lab_img, CV_32FC3, 1.0f / 255.0f);
	cv::cvtColor(temp_lab_img, temp_lab_img, CV_BGR2Lab);

	cv::Mat result;
	for (int i = 0; i < 10; i++)
	{
		bilateralFilter(temp_lab_img, result, -1, 7, 7);
		result.copyTo(temp_lab_img);
	}

	return result;
}

/**
* \brief
* \param img
* \param omega
* \return
*/
std::vector<cv::Vec3f> compute_feature_vectors(const cv::Mat& img, const std::vector<cv::Point>& omega)
{
	std::vector<cv::Vec3f> result;

	for (auto it = omega.begin(); it != omega.end(); it++)
	{
		result.push_back(img.at<cv::Vec3f>(*it));
	}

	return result;
}

/**
* \brief
* \param img
* \param feature_vectors
* \return
*/
cv::Mat compute_DP(const cv::Mat& img, std::vector<cv::Vec3f> feature_vectors)
{
	cv::Mat dp(img.rows, img.cols, CV_32FC1);

	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			float resultnorm = 0.0f;

			for (int k = 0; k < feature_vectors.size(); k++) {
				resultnorm += cv::norm(img.at<cv::Vec3f>(i, j) - feature_vectors[k]);
			}

			dp.at<float>(i, j) = resultnorm;
		}
	}

	return dp;
}

/**
* \brief
* \param img
* \return
*/
cv::Mat compute_histogram(const cv::Mat& img)
{
	// Params for calcHist
	int hist_size = 256;
	float range[] = { 0, 256 };
	const float* hist_range = { range };
	bool uniform = true;
	bool accumulate = false;


	// Compute
	cv::Mat histogram;
	cv::calcHist(&img, 1, 0, cv::Mat(), histogram, 1, &hist_size, &hist_range, uniform, accumulate);

	// Draw the histogram
	int hist_w = 512; int hist_h = 400;
	int bin_w = cvRound((double)hist_w / hist_size);

	cv::Mat hist_image(hist_h, hist_w, CV_8UC3, cv::Scalar(0, 0, 0));

	cv::normalize(histogram, histogram, 0, hist_image.rows, cv::NORM_MINMAX, -1, cv::Mat());

	for (int i = 1; i < hist_size; i++)
	{
		line(hist_image, cv::Point(bin_w*(i - 1), hist_h - cvRound(histogram.at<float>(i - 1))),
			cv::Point(bin_w*(i), hist_h - cvRound(histogram.at<float>(i))),
			cv::Scalar(255, 255, 255), 2, 8, 0);;
	}

	return hist_image;
}

/**
* \brief
* \param img
* \param points_number
* \return
*/
std::vector<cv::Point> sample_points(const cv::Mat& img, int points_number)
{
	cv::Mat norm_img;
	cv::normalize(img, norm_img, 0, 255, cv::NORM_MINMAX, CV_32FC1);

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

	// Put current system clock as random generator seed
	srand(clock());

	std::vector<cv::Point> result;
	for (int p = 0; p < points_number; p++)
	{
		// Generate a random number and find where it falls in the
		// rows probabilities

		float rand_row = rand() / static_cast<float>(RAND_MAX);
		int drawn_row_index = 0;

		for (int i = 0; i < img.rows; i++)
		{
			if (row_cumulated_probs[i] <= rand_row && rand_row <= row_cumulated_probs[i + 1])
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

		result.push_back(cv::Point(drawn_col_index, drawn_row_index));
	}

	return result;
}

/**
* \brief
* \param src_img
* \param points_to_weather
* \param weathered_patch
* \param weathering_map
* \return
*/
cv::Mat weather_points(const cv::Mat& src_img, const std::vector<cv::Point>& points_to_weather, const cv::Mat& weathered_patch, const cv::Mat& weathering_map)
{
	cv::Mat weathered_img(src_img.rows, src_img.cols, CV_32FC3);
	src_img.copyTo(weathered_img);

	cv::Mat normalized_weathering_map;
	cv::normalize(weathering_map, normalized_weathering_map, 0, 1, cv::NORM_MINMAX, CV_32FC1);

	for (int p = 0; p < points_to_weather.size(); p++)
	{
		cv::Point current_point = points_to_weather[p];
		int patch_x = std::floor(current_point.x - weathered_patch.cols / 2);
		int patch_y = std::floor(current_point.y - weathered_patch.rows / 2);;

		cv::Point patch_point(patch_x, patch_y);
		for (int i = 0; i < weathered_patch.cols; i++)
		{
			patch_point.y = patch_y;

			for (int j = 0; j < weathered_patch.rows; j++)
			{
				if (!(patch_point.x < 0 || patch_point.x >= src_img.cols || patch_point.y < 0 || patch_point.y >= src_img.rows))
				{
					weathered_img.at<cv::Vec3f>(patch_point) = (1 - normalized_weathering_map.at<float>(patch_point)) * src_img.at<cv::Vec3f>(patch_point) +
						normalized_weathering_map.at<float>(patch_point) * weathered_patch.at<cv::Vec3f>(cv::Point(i, j));
				}

				patch_point.y++;
			}

			patch_point.x++;
		}
	}

	return weathered_img;
}

/**
* \brief
* \param probability_map
* \param weathered_points
*/
void update_probability_map(cv::Mat& probability_map, const std::vector<cv::Point>& weathered_points)
{
	for (int i = 0; i < weathered_points.size(); i++)
	{
		probability_map.at<unsigned char>(weathered_points[i]) = 0;
	}
}

/**
* \brief
* \param event
* \param x
* \param y
*/
void mouseCallback(int event, int x, int y, int, void*)
{
	// Check for left mouse click
	if (event == cv::EVENT_LBUTTONDOWN)
	{
		// Add a new point to the list
		cv::Point new_point(x, y);
		m_selected_points.push_back(new_point);

		// Show the point on the source image
		std::cout << "Point " << m_selected_points.size() << " : " << new_point << " in the image carthesian coordinate system" << std::endl;

		cv::drawMarker(m_src_img_display, new_point, cv::Scalar(0, 0, 255));
		cv::imshow("Source Image", m_src_img_display);
		cv::waitKey(1);

		if (m_selected_points.size() >= 2) {
			// Disable mouse callback on source image window			
			cv::setMouseCallback("Source Image", nullptr, nullptr);

			// Extract the region around the second weathered_patch to constiute
			// the wetathered weathered_patch
			cv::Rect weathered_patch_region = cv::Rect(std::floor(m_selected_points[1].x - m_patch_size / 2), std::floor(m_selected_points[1].y - m_patch_size / 2), m_patch_size, m_patch_size);
			cv::Mat weathered_patch = m_src_img(weathered_patch_region);
			weathered_patch.convertTo(weathered_patch, CV_32FC3);

			// Show the region on the source image
			cv::rectangle(m_src_img_display, weathered_patch_region.tl(), weathered_patch_region.br(), cv::Scalar(0, 0, 255));
			cv::imshow("Source Image", m_src_img_display);
			cv::waitKey(1);
			std::cout << "\nBegin processing..." << std::endl;

			std::cout << "----- Compute Omega set" << std::endl;

			std::vector<cv::Point> omega = compute_omega(m_src_img, m_selected_points[0], m_omega_radius);

			// Display the set
			for (int i = 0; i < omega.size(); i++)
			{
				m_src_img_display.at<cv::Vec3b>(omega[i])[0] = 255;
				m_src_img_display.at<cv::Vec3b>(omega[i])[1] = 255;
			}
			cv::imshow("Source Image", m_src_img_display);
			cv::waitKey(1);

			std::cout << "----- Apply preprocessing" << std::endl;
			cv::Mat preprocessed_src_img = preprocess(m_src_img);

			std::cout << "----- Compute feature vectors" << std::endl;
			std::vector<cv::Vec3f> feature_vectors = compute_feature_vectors(preprocessed_src_img, omega);

			std::cout << "----- Compute DP" << std::endl;
			cv::Mat dp = compute_DP(preprocessed_src_img, feature_vectors);

			// Show DP
			cv::Mat dp_display;
			cv::normalize(dp, dp_display, 0, 255, cv::NORM_MINMAX, CV_8UC1);
			cv::namedWindow("DP Map", cv::WINDOW_AUTOSIZE);
			cv::imshow("DP Map", dp_display);
			cv::waitKey(1);

			std::cout << "----- Compute Probability Map" << std::endl;
			cv::Mat probability_map = 1.0f - dp;

			// Normalize for easier manipulation and visualization
			cv::normalize(probability_map, probability_map, 0, 255, cv::NORM_MINMAX, CV_8UC1);
			cv::threshold(probability_map, probability_map, 170, 255, cv::THRESH_TOZERO); // #FIXME: Find a way to compute the threshold automatically

																						  // Show probability map
			cv::namedWindow("Probability Map", cv::WINDOW_AUTOSIZE);
			cv::imshow("Probability Map", probability_map);
			cv::waitKey(1);

			// Compute probability map histogram
			cv::Mat probability_map_histogram = compute_histogram(probability_map);

			// Show probability map histogram
			cv::namedWindow("Probability Map Histogram", cv::WINDOW_AUTOSIZE);
			cv::imshow("Probability Map Histogram", probability_map_histogram);
			cv::waitKey(1);

			std::cout << "----- Weather image" << std::endl;

			// Prepare result window
			cv::namedWindow("Weathered Image", cv::WINDOW_AUTOSIZE);

			cv::Mat previous_weathered_img;
			m_src_img.convertTo(previous_weathered_img, CV_32FC3);

			for (int i = 1; i <= m_iterations; i++)
			{
				std::cout << "---------- Progress : " << std::to_string(int(i * 100 / m_iterations)) << "%\r" << std::flush;

				std::vector<cv::Point> sampled_points = sample_points(probability_map, m_samples);

				// Show the points which are to be weathered
				for (int j = 0; j < sampled_points.size(); j++)
				{
					cv::drawMarker(m_src_img_display, sampled_points[j], cv::Scalar(255, 255, 255), 0, 5);
				}
				cv::imshow("Source Image", m_src_img_display);
				cv::waitKey(1);

				// Blend with patch
				cv::Mat new_weathered_img = weather_points(previous_weathered_img, sampled_points, weathered_patch, dp);

				// Show updated result
				cv::Mat weathered_img_display;
				new_weathered_img.convertTo(weathered_img_display, CV_8UC3);
				cv::imshow("Weathered Image", weathered_img_display);
				cv::waitKey(1);

				// Nullify the probability of already weathered points
				update_probability_map(probability_map, sampled_points);

				// Show updated probability map
				cv::normalize(probability_map, probability_map, 0, 255, cv::NORM_MINMAX, CV_8UC1);
				cv::imshow("Probability Map", probability_map);
				cv::waitKey(1);

				// Update and show probability map histogram
				cv::Mat probability_map_histogram = compute_histogram(probability_map);
				cv::imshow("Probability Map Histogram", probability_map_histogram);
				cv::waitKey(1);

				// Replace previously weathered image by the new one
				new_weathered_img.copyTo(previous_weathered_img);
			}

			std::cout << "\nFinished processing..." << std::endl;
			std::cout << "\nPress \"Escape\" while focused on Source Image window to exit" << std::endl;
		}
	}
}

/**
* \brief
* \param new_value
*/
void on_patch_size_change(int new_value, void*)
{
	m_patch_size = new_value;
}

/**
* \brief
* \param new_value
*/
void on_omega_radius_change(int new_value, void*)
{
	m_omega_radius = new_value;
}

/**
* \brief
* \param new_value
*/
void on_iterations_change(int new_value, void*)
{
	m_iterations = new_value;
}

/**
* \brief
* \param new_value
*/
void on_samples_change(int new_value, void*)
{
	m_samples = new_value;
}

int main() {
	// Load input image
	char* src_img_path = "../test1.jpg"; // #TODO: Pass through program argument
	m_src_img = cv::imread(src_img_path);
	m_src_img.copyTo(m_src_img_display);

	// Create source image window
	cv::namedWindow("Source Image", cv::WINDOW_AUTOSIZE);

	// Add parameters trackbars ti the window
	cv::createTrackbar("Patch size", "Source Image", &m_patch_size, MAX_PATCH_SIZE, on_patch_size_change);
	cv::createTrackbar("Omega radius", "Source Image", &m_omega_radius, MAX_OMEGA_RADIUS, on_omega_radius_change);
	cv::createTrackbar("Iterations", "Source Image", &m_iterations, MAX_ITERATIONS, on_iterations_change);
	cv::createTrackbar("Samples", "Source Image", &m_samples, MAX_SAMPLES, on_samples_change);

	// Display source image
	cv::imshow("Source Image", m_src_img);
	cv::waitKey(1);

	// Set our custom callback function on the source image windoow
	cv::setMouseCallback("Source Image", mouseCallback, nullptr);

	std::cout << "Click on two points on the image : " << std::endl;
	std::cout << "1 - The first one on a clean area" << std::endl;
	std::cout << "2 - The second one on a weathered area\n" << std::endl;

	// Exit loop
	while (true)
	{
		// Press Esc to exit program
		int key = cvWaitKey(10);
		if (key == 27) break;
	}

	return 0;
}