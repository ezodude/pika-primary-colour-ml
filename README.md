#######################################################################
#######################   Overview   ################################## 
#######################################################################

We created a dataset containing roughly 80 labelled pictures. A neural network was trained to predict the color for tiles of the image but we found that
The neural network can not handle different tile sizes and 
The accuracy of the neural network only reaches 80 to 85%. This can be solved by
Using a larger dataset
Using transfer learning from similar datasets. 
Because of those limitations and an approaching deadline we evaluated a more classical approach that uses a statistical analysis. This approach was implemented in Python and then translated into C++. The code was tested and the accuracy surpassed 90%. The runtime of the final algorithm is below 0.1 seconds for a 100 by 100 pixel tile.

We will now describe the components in more detail.

#######################################################################
##############  Generating the model file (Python) ####################
#######################################################################

The first step is to generate the model (json) file. This is done in Python with a call to the compute_color_mapping.py script. There are three parameters that can be specified:

Usage: python3 compute_color_mapping.py [options]

Options:
  -h, --help            show this help message and exit
  --data_set=DATA_SET   Location of the data set.
  --color_grouping=COLOR_GROUPING
                        How big should the clusters of colors be that willbe
                        grouped together.
  --output_directory=OUTPUT_DIRECTORY
                        Where should we output the files to.

The path to the dataset, for example C:\Users\Pika\Dataset\. The dataset folder should contain one folder called “Training”. All files in that folder will be used to generate the model file. Every file in that folder should be in the png format and come with an additional file with the _mask.png postfix that includes the desired colouring. This mask file has to have exactly the same dimensions as the original file!

The second parameter is color_grouping. For now this should either not be used or set to 5. Other values are supported by the python script but not by the C++ classifier so do not modify it unless you also adapt the C++ code. 
The last parameter is the output_directory. This will be the directory where the json file is placed. If none is specified the current directory will be used. 

The script will run for about an hour, outputting the current file it processes. When it processed all files it will output values from 0 to 255. After that the file should be generated. You can expect a file size around 2MB. A typical file should look something like this:

[[784897, 2047, 2133, 2646], [209430, 214, 1152, 748], [155039, 379, 1555,... and so on. 

We are now ready to use the classifier in C++.

#######################################################################
#################### Using the classifier ############################# 
#######################################################################

The classifier has the following interface:

public:
	ColorClassifier(const std::string& load_from_object);
	inline int compute_idx(const cv::Vec3b & pixel);
	std::tuple<double, double, double, double> compute_percentage(const cv::Mat & tile);
	bool is_blue(const cv::Mat & tile, float alpha);
	bool is_red(const cv::Mat & tile, float alpha);
	bool is_yellow(const cv::Mat & tile, float alpha);
	~ColorClassifier();
private:
	std::vector<float> p_none;
	std::vector<float> p_blue;
	std::vector<float> p_red;
	std::vector<float> p_yellow;
	const int STEP_SIZE = 5;
	const int BASE = 255 / STEP_SIZE + 1;

It can be constructed by providing the path to the json file. Please make sure this is the correct path, otherwise the classifier might be initialized incorrectly. 

There are 3 functions is_blue, is_red, is_yellow that can be used to classify a tile. Alpha is a parameter that can be fine tuned. A lower value will generate more false positives a higher value will generate more false negatives. A recommended value is 0.18 (slightly more false positives). 

The first parameter is an openCV Matrix in the RGB format. Make sure the color format is correct because openCV loads image files in the BGR format. In that case you need to transform the file first with cvtColor(image, image, COLOR_BGR2RGB); 
The Utils.cpp file contains quite a few examples how to read and process a file. If you only want to classify a tile, select the tile like this image.rowRange(i, to_i).colRange(j, to_j) and give the resulting sub matrix to the function.

The library also includes a compute_percentage function that will return a tuple containing 4 values. These are the raw (not normalized) percentages for none, blue, red and yellow respectively. You can see an example use case of that function inside the is_blue/red/yellow function. 

The class also contains a private variable STEP_SIZE. You can adapt that step size if you want to rerun the python script with a different COLOR_GROUPING. You should not be required to do this and stuff might break.

Lastly compute_idx computes an index that is used to look up the percentages for one specific pixel. This is more of an internal function but if you just want to classify individual pixels you can use it by giving a pixel in the RGB format. It will compute the lookup position and you can write an internal function that turns that index into a percentage by looking into the correct table (p_red, p_blue, p_yellow). 
