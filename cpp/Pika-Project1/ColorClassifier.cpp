#include "ColorClassifier.h"
#include <opencv2/core/core.hpp>
#include "Utils.h"
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iterator>
#include <Windows.h>
#include <numeric>
#include <tuple>


using namespace std;

ColorClassifier::ColorClassifier(const string& load_from_object)
{
	vector<vector<int>> parsed_content;
	ifstream file(load_from_object);
	if (file.is_open())
	{

		// We expect a pattern like [[584859, 2064, 2133, 2439], [201442, 210, 1152, 599], ...
		// This will generate tokens: "[[584859,", "2064,", "2133,", "2439],"...
		// We can ignore everything that is not a digit and assume the groups are always of size 4.
		int counter = 0;
		int sub_counter = 0;
		string current_content;
		while (file >> current_content) {
			if (counter % 4 == 0) {
				// We start a new element.
				sub_counter = 0;
				parsed_content.push_back(vector<int>(4, 0));
			}

			// Extract only the digits:
			string digits = "";
			for (char c : current_content) {
				if (isdigit(c)) {
					digits += c;
				}
			}
			// Save the current number.
			parsed_content.back()[sub_counter] = stoi(digits);
			sub_counter++;
			counter++;
		}

	}
	else {
		throw exception("Could not open file!");
	}

	// From the parsed_content we have to create maps for every color
	for (const vector<int>& v : parsed_content) {
		long total = accumulate(v.begin(), v.end(), 0);
		if (total == 0) {
			// TODO: Ideally we want to expand the training set or look for the closest color
			// that we found. But for now we just assume this color is not interesting for us.
			p_none.push_back(0);
			p_blue.push_back(0);
			p_red.push_back(0);
			p_yellow.push_back(0);
		}
		else {
			assert(total > 0);
			p_none.push_back(v[0] / float(total));
			p_blue.push_back(v[1] / float(total));
			p_red.push_back(v[2] / float(total));
			p_yellow.push_back(v[3] / float(total));
		}
	}

}

inline int ColorClassifier::compute_idx(const cv::Vec3b& pixel) {
	int r, g, b;
	r = pixel[0];
	g = pixel[1];
	b = pixel[2];
	return int(r / STEP_SIZE) + BASE * int(g / STEP_SIZE) + BASE * BASE * int(b / STEP_SIZE);
}


tuple<double, double, double, double> ColorClassifier::compute_percentage(const cv::Mat& patch, float alpha) {
	double acc_p_none = 0;
	double acc_p_blue = 0;
	double acc_p_red = 0;
	double acc_p_yellow = 0;

	for (int i = 0; i < patch.rows; ++i) {
		const cv::Vec3b* pixel = patch.ptr<cv::Vec3b>(i);
		for (int j = 0; j < patch.cols; ++j) {
			int idx = compute_idx(pixel[j]);
			acc_p_none = acc_p_none + p_none[idx];
			acc_p_blue = acc_p_blue + p_blue[idx];
			acc_p_red = acc_p_red + p_red[idx];
			acc_p_yellow = acc_p_yellow + p_yellow[idx];
		}
	}
	acc_p_none = acc_p_none* alpha;
	return make_tuple(acc_p_none, acc_p_blue, acc_p_red, acc_p_yellow);
}

bool ColorClassifier::is_blue(const cv::Mat& patch, float alpha) {
	tuple<double, double, double, double> acc_p_tuple = compute_percentage(patch, alpha);
	double acc_p_none = std::get<0>(acc_p_tuple);
	double acc_p_blue = std::get<1>(acc_p_tuple);
	double acc_p_red = std::get<2>(acc_p_tuple);
	double acc_p_yellow = std::get<3>(acc_p_tuple);
	return (acc_p_blue > acc_p_none) && (acc_p_blue >= acc_p_red) && (acc_p_blue >= acc_p_yellow);
}

bool ColorClassifier::is_red(const cv::Mat& patch, float alpha) {
	tuple<double, double, double, double> acc_p_tuple = compute_percentage(patch, alpha);
	double acc_p_none = std::get<0>(acc_p_tuple);
	double acc_p_blue = std::get<1>(acc_p_tuple);
	double acc_p_red = std::get<2>(acc_p_tuple);
	double acc_p_yellow = std::get<3>(acc_p_tuple);
	return (acc_p_red > acc_p_none) && (acc_p_red >= acc_p_red) && (acc_p_red >= acc_p_yellow);
}

bool ColorClassifier::is_yellow(const cv::Mat& patch, float alpha) {
	tuple<double, double, double, double> acc_p_tuple = compute_percentage(patch, alpha);
	double acc_p_none = std::get<0>(acc_p_tuple);
	double acc_p_blue = std::get<1>(acc_p_tuple);
	double acc_p_red = std::get<2>(acc_p_tuple);
	double acc_p_yellow = std::get<3>(acc_p_tuple);
	return (acc_p_yellow > acc_p_none) && (acc_p_yellow >= acc_p_blue) && (acc_p_yellow >= acc_p_red);
}

ColorClassifier::ColorClassifier(const vector<string>& training_files)
{
	// Currently not supported.
}

ColorClassifier::~ColorClassifier()
{
}
