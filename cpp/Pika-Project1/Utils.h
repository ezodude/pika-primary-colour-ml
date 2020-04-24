#pragma once

#include <opencv2/core/core.hpp>

class Utils
{
public:
	Utils();
	~Utils();

	template<typename Out>
	static void split(const std::string &s, char delim, Out result);
	static std::vector<std::string> split(const std::string & s, char delim);
	static std::string join_path(const std::string & part1, const std::string & part2);
	static std::vector<std::string> list_files_in_folder(std::string & folder, bool(*filter)(const std::string &));
	static bool string_ends_with(std::string const & value, std::string const & ending);
	static std::pair<cv::Mat, cv::Mat> read_complete_file(const std::string & folder, const std::string & file_name);
	static cv::Mat encode_label_image(const cv::Mat & label_image);
	static std::vector<std::vector<std::vector<unsigned char>>> mat2vec(const cv::Mat & input);
};

