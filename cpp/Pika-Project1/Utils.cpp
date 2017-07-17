#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iterator>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <Windows.h>

#include "Utils.h"



using namespace cv;
using namespace std;

Utils::Utils()
{
}

template<typename Out>
void Utils::split(const string &s, char delim, Out result) {
	stringstream ss;
	ss.str(s);
	string item;
	while (getline(ss, item, delim)) {
		*(result++) = item;
	}
}

vector<string> Utils::split(const string &s, char delim) {
	vector<string> elems;
	Utils::split(s, delim, back_inserter(elems));
	return elems;
}

string Utils::join_path(const string& part1, const string& part2) {
	const char delimeter = '\\';
	if (part1.back() == delimeter) {
		return part1 + part2;
	}
	return part1 + delimeter + part2;
}

vector<string> Utils::list_files_in_folder(string& folder, bool(*filter)(const string&))
{
	vector<string> names;
	string search_path = join_path(folder, "*.*");
	WIN32_FIND_DATA fd;
	HANDLE hFind = ::FindFirstFile(search_path.c_str(), &fd);
	if (hFind != INVALID_HANDLE_VALUE) {
		do {
			// Read all (real) files in current folder:
			if (!(fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) {
				if (filter(fd.cFileName)) {
					names.push_back(fd.cFileName);
				}
			}
		} while (::FindNextFile(hFind, &fd));
		::FindClose(hFind);
	}
	return names;
}

bool Utils::string_ends_with(string const & value, string const & ending)
{
	if (ending.size() > value.size()) return false;
	return equal(ending.rbegin(), ending.rend(), value.rbegin());
}

pair<Mat, Mat> Utils::read_complete_file(const string& folder, const string& file_name) {
	//Read the image.
	string file_name_reduced = Utils::join_path(folder, file_name);
	file_name_reduced = file_name_reduced.substr(0, file_name_reduced.length() - 4);
	cout << file_name_reduced << endl;

	Mat image = imread(file_name_reduced + ".png", CV_LOAD_IMAGE_COLOR);
	cvtColor(image, image, COLOR_BGR2RGB);

	int h = image.size[0];
	int w = image.size[1];

	double scaling_factor = 0;
	if (h > w) {
		scaling_factor = 1280 / float(w);
	} else {
		scaling_factor = 1280 / float(h);
	}

	//Read the mask / label image.
	Mat label_image = imread(file_name_reduced + "_mask.png", CV_LOAD_IMAGE_COLOR);
	cvtColor(label_image, label_image, COLOR_BGR2RGB);

	if (scaling_factor > 1) {
		resize(image, image, Size(), scaling_factor, scaling_factor, CV_INTER_CUBIC);
		resize(label_image, label_image, Size(), scaling_factor, scaling_factor, CV_INTER_CUBIC);
	}
	else {
		resize(image, image, Size(), scaling_factor, scaling_factor, CV_INTER_AREA);
		resize(label_image, label_image, Size(), scaling_factor, scaling_factor, CV_INTER_AREA);
	}

	return make_pair(image, label_image);
}


// Converts the label image into actual values between[0..4]
Mat Utils::encode_label_image(const Mat& label_image) {

	int h = label_image.size[0];
	int w = label_image.size[1];

	int num_classes = 4;
	Mat mapping(num_classes, 1, CV_8UC3);
	mapping.at<Vec3b>(0, 0) = Vec3b(0, 0, 0);  // Background
	mapping.at<Vec3b>(1, 0) = Vec3b(0, 0, 255);  // Blue
	mapping.at<Vec3b>(2, 0) = Vec3b(255, 0, 0);  // Red
	mapping.at<Vec3b>(3, 0) = Vec3b(255, 255, 0);  // Yellow

	Mat labels(h, w, CV_8U);
	for (int i = 0; i < h; ++i) {
		for (int j = 0; j < w; ++j) {
			unsigned char closest = 0;
			double min_distance = 10000;
			for (unsigned char k = 0; k < num_classes; ++k) {
				Vec3b d = label_image.at<Vec3b>(i, j) - mapping.at<Vec3b>(k, 0);
				double distance = norm(d);
				if (distance < min_distance) {
					closest = k;
					min_distance = distance;
				}
			}
			labels.at<unsigned char>(i, j) = closest;
		}
	}
	return labels;
}


Utils::~Utils()
{
}
