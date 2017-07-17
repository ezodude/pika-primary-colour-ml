#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/core/utility.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>
#include <Windows.h>

#include "ColorClassifier.h"
#include "Utils.h"

using namespace cv;
using namespace std;


vector<int> count_occurences(const Mat& patch) {
	vector<int> result(4, 0);
	for (int i = 0; i < patch.rows; ++i) {
		const unsigned char* pixel = patch.ptr<unsigned char>(i);
		for (int j = 0; j < patch.cols; ++j) {
			result[pixel[j]]++;
		}
	}
	return result;
}

int main()
{
	string dataset = "C:\\Users\\sylvus\\Startups\\HighDimension\\Pika\\Dataset\\";
	auto valid_file = [](const string& file) { return Utils::string_ends_with(file, ".png") && !Utils::string_ends_with(file, "_mask.png"); };

	string training_folder = Utils::join_path(dataset, "Training");
	vector<string> training_files = Utils::list_files_in_folder(training_folder, valid_file);

	string validation_folder = Utils::join_path(dataset, "Validating");
	vector<string> validation_files = Utils::list_files_in_folder(validation_folder, valid_file);

	ColorClassifier cc("C:\\Users\\sylvus\\Startups\\HighDimension\\Pika\\Pika-Project\\Project1\\color_statistic.json");


	int total_count_blue_correct = 0;
	int total_count_blue_all = 0;
	int total_count_red_correct = 0;
	int total_count_red_all = 0;
	int total_count_yellow_correct = 0;
	int total_count_yellow_all = 0;
	int total_count_none_correct = 0;
	int total_count_none_all = 0;

	for (string file : training_files) {
		pair<Mat, Mat> sample = Utils::read_complete_file(training_folder, file);
		Mat label = Utils::encode_label_image(sample.second);

		int count_blue_correct = 0;
		int count_blue_all = 0;
		int count_red_correct = 0;
		int count_red_all = 0;
		int count_yellow_correct = 0;
		int count_yellow_all = 0;
		int count_none_correct = 0;
		int count_none_all = 0;

		int chunk_size = 128;
		int h = sample.first.size[0];
		int w = sample.first.size[1];
		Mat label_reduced(h / chunk_size + 1, w / chunk_size + 1, CV_8U);
		Mat predicted_reduced(h / chunk_size + 1, w / chunk_size + 1, CV_8U);
		for (int i = 0; i < h - 1; i = i + chunk_size) {
			for (int j = 0; j < w - 1; j = j + chunk_size) {
				int to_i = min(i + chunk_size, h - 1);
				int to_j = min(j + chunk_size, w - 1);
				const Mat& patch = sample.first.rowRange(i, to_i).colRange(j, to_j);
				bool blue = cc.is_blue(patch, 0.18);
				bool red = cc.is_red(patch, 0.18);
				bool yellow = cc.is_yellow(patch, 0.18);

				// Report our findings for visual feedback.
				int color = 0;
				if (blue) {
					color = 1;
				}
				if (red) {
					color = 2;
				}
				if (yellow) {
					color = 3;
				}
				predicted_reduced.at<unsigned char>(i / chunk_size, j / chunk_size) = color;

				vector<int> oc = count_occurences(label.rowRange(i, to_i).colRange(j, to_j));

				label_reduced.at<unsigned char>(i / chunk_size, j / chunk_size) = 0;
				if ((oc[1] > oc[0] * 0.4) && (oc[1] > oc[2]) && (oc[1] > oc[3])) {
					// Should be blue:
					if (blue) {
						count_blue_correct += 1;
					}
					count_blue_all += 1;
					label_reduced.at<unsigned char>(i / chunk_size, j / chunk_size) = 1;
				}
				if ((oc[2] > oc[0] * 0.4) && (oc[2] > oc[1]) && (oc[2] > oc[3])) {
					// Should be red:
					if (red) {
						count_red_correct += 1;
					}
					count_red_all += 1;
					label_reduced.at<unsigned char>(i / chunk_size, j / chunk_size) = 2;
				}
				if ((oc[3] > oc[0] * 0.4) && (oc[3] > oc[1]) && (oc[3] > oc[2])) {
					// Should be yellow:
					if (yellow) {
						count_yellow_correct += 1;
					}
					count_yellow_all += 1;
					label_reduced.at<unsigned char>(i / chunk_size, j / chunk_size) = 3;
				}
				if ((oc[0] * 0.4 > oc[1]) && (oc[0] * 0.4 > oc[2]) && (oc[0] * 0.4 > oc[3])) {
					// Should be black:
					if (!(blue || red || yellow)) {
						count_none_correct += 1;
					}
					count_none_all += 1;
				}
			}

		}

		cout << "Blue: " << count_blue_correct / float(count_blue_all) << endl;
		cout << "Red: " << count_red_correct / float(count_red_all) << endl;
		cout << "Yellow: " << count_yellow_correct / float(count_yellow_all) << endl;
		cout << "None: " << count_none_correct / float(count_none_all) << endl;

		total_count_blue_correct += count_blue_correct;
		total_count_blue_all += count_blue_all;
		total_count_red_correct += count_red_correct;
		total_count_red_all += count_red_all;
		total_count_yellow_correct += count_yellow_correct;
		total_count_yellow_all += count_yellow_all;
		total_count_none_correct += count_none_correct;
		total_count_none_all += count_none_all;

		if (true) {
			namedWindow("Display window original", WINDOW_NORMAL);
			Mat image;
			cvtColor(sample.first, image, COLOR_RGB2BGR);
			imshow("Display window original", image);
			resizeWindow("Display window original", 600, 600);
			namedWindow("Display window predicted", WINDOW_NORMAL);
			imshow("Display window predicted", (255 / 3)*predicted_reduced);
			resizeWindow("Display window predicted", 600, 600);
			namedWindow("Display window label", WINDOW_NORMAL);
			imshow("Display window label", (255 / 3)*label_reduced);
			resizeWindow("Display window label", 600, 600);
			waitKey(0);
		}
	}

	cout << "Total Blue: " << total_count_blue_correct / float(total_count_blue_all) << endl;
	cout << "Total Red: " << total_count_red_correct / float(total_count_red_all) << endl;
	cout << "Total Yellow: " << total_count_yellow_correct / float(total_count_yellow_all) << endl;
	cout << "Total None: " << total_count_none_correct / float(total_count_none_all) << endl;

	float avg1 = total_count_blue_correct / float(total_count_blue_all) + total_count_red_correct / float(total_count_red_all) + total_count_yellow_correct / float(total_count_yellow_all) + total_count_none_correct / float(total_count_none_all);
	cout << "Average 1: " << 0.25*avg1 << endl;

	float avg2 = (total_count_blue_correct + total_count_red_correct + total_count_yellow_correct  + total_count_none_correct) / float(total_count_blue_all + total_count_red_all + total_count_yellow_all + total_count_none_all);
	cout << "Average 2: " << avg2 << endl;


	string temp;
	cin >> temp;
	return 0;
}