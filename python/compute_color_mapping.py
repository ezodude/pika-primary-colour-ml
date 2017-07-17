import json
import os
from optparse import OptionParser

import numpy as np
from skimage.io import imread


def encode_label_image(label_image):
    """
    Encodes the label image with values in range [0..4] corresponding to:
    0: Background
    1: Blue
    2: Red
    3: Yellow

    Parameters
    ----------
    label_image: The original label image

    Returns
    -------
    A new 2D array with values between [0..4] corresponding to
    the correct class.

    """
    (h, w) = label_image.shape[:2]
    reshaped = label_image.reshape((h * w, 3))

    num_classes = 4
    mapping = np.zeros(shape=(num_classes, 1, 3), dtype=np.uint8)
    mapping[0, 0, :] = [0, 0, 0]  # Background
    mapping[1, 0, :] = [0, 0, 255]  # Blue
    mapping[2, 0, :] = [255, 0, 0]  # Red
    mapping[3, 0, :] = [255, 255, 0]  # Yellow

    distances = np.sqrt(((reshaped - mapping) ** 2).sum(axis=2))
    labels = np.argmin(distances, axis=0)
    labels = labels.reshape((h, w))
    return labels


def read_complete_file(dataset, file_name):
    """

    Parameters
    ----------
    dataset: The folder containing the dataset.
    file_name: The filename of the file we want to read.

    Returns
    -------
    The image and the label image.

    """
    # Read the image.
    image = imread(os.path.join(dataset, file_name) + ".png")
    # We only want the RGB channels. No alpha!
    image = image[:, :, 0:3]

    # Read the mask/label image.
    label_image = imread(os.path.join(dataset, file_name) + "_mask.png")
    # We only want the RGB channels. No alpha!
    label_image = label_image[:, :, 0:3]
    return image, label_image


def record_color(color_count, image, label_image):
    """
    Collects all the color to label mappings for a given image
    Parameters
    ----------
    color_count: The mapping we need to update.
    image: The image file
    label_image: The labels in range [0..4]

    Returns
    -------
    Nothing.

    """
    (h, w) = image.shape[:2]
    for i in range(h):
        for j in range(w):
            t = tuple(image[i, j])
            if t in color_count:
                color_count[t][label_image[i, j]] += 1
            else:
                color_count[t] = [0, 0, 0, 0]


def main():
    """
    Run this file to recompute the statistics that are required for the
    C++ real time classifier.
    Returns
    -------
    Nothing, but saves the required files for the C++ classifier.
    """
    parser = OptionParser()
    parser.add_option("--data_set", action="store", type="string",
                      dest="data_set",
                      help="Location of the data set.",
                      default="C:\\Users\\sylvus\\Startups\\"
                              "HighDimension\\Pika\\Dataset\\")
    parser.add_option("--color_grouping", action="store", type="int",
                      dest="color_grouping",
                      help="How big should the clusters of colors be that will"
                           "be grouped together.",
                      default=5)
    parser.add_option("--output_directory", action="store", type="string",
                      dest="output_directory",
                      help="Where should we output the files to.", default="")
    (options, _) = parser.parse_args()

    training_files = []
    training_folder = os.path.join(options.data_set, "Training")
    for file_name in os.listdir(training_folder):
        if file_name.endswith(".png") and not file_name.endswith("_mask.png"):
            training_files.append(file_name[0:-4])

    print("Computing initial color count.")
    color_count = {}
    for f in training_files:
        print("Loading file: ", f)
        image, label_image = read_complete_file(training_folder, f)
        label_image = encode_label_image(label_image)
        record_color(color_count, image, label_image)

    step_size = options.color_grouping
    base = int(255 / step_size) + 1
    color_count_grouped = [[0, 0, 0, 0] for _ in
                           range(base + base ** 2 + base ** 3 + 1)]

    for r in range(0, 256, step_size):
        print("Computing groupings out of 256: ", r)
        for g in range(0, 256, step_size):
            for b in range(0, 256, step_size):
                # We have a cube with side length step_size,
                # let us combine all those values into one.
                total_count = [0, 0, 0, 0]
                for i in range(step_size):
                    for j in range(step_size):
                        for k in range(step_size):
                            current_values = (r + i, g + j, b + k)
                            if current_values not in color_count:
                                continue
                            c_count = color_count[current_values]
                            for sub_i in range(4):
                                total_count[sub_i] += c_count[sub_i]
                color_count_grouped[
                    int(r / step_size +
                        base * g / step_size +
                        base ** 2 * b / step_size)] = total_count

    output_file = 'color_statistic.json'
    if len(options.output_directory) > 0:
        output_file = os.path.join(options.output_directory, output_file)
    with open(output_file, 'w') as f:
        json.dump(color_count_grouped, f, ensure_ascii=False)


if __name__ == '__main__':
    main()
