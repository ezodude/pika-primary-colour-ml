import json
import os
from optparse import OptionParser

import numpy as np
from skimage.io import imread

import matplotlib.pyplot as plt

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


def segment_by_color(image, color_map, options):
    step_size = options.color_grouping
    base = int(255 / step_size) + 1

    (h, w) = image.shape[:2]
    new_image = np.zeros(shape=(h, w), dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            t = image[i, j] / step_size
            distr = color_map[
                int(t[0]) + base * int(t[1]) + base ** 2 * int(t[2])]
            new_image[i, j] = np.argmax(distr)
    return new_image


def make_color(image, color_map, options):
    step_size = options.color_grouping
    base = int(255 / step_size) + 1

    (h, w) = image.shape[:2]
    percentage = np.zeros(shape=4, dtype=np.float32)
    counter = 0
    for i in range(h):
        for j in range(w):
            t = image[i, j] / step_size
            distr = color_map[
                int(t[0]) + base * int(t[1]) + base ** 2 * int(t[2])]

            total = np.sum(distr)
            if total > 0:
                percentage += (np.array(distr)/float(total))
            counter += 1

    #print(percentage)
    percentage[0] *= 0.1
    color = np.argmax(percentage)

    new_image = np.ones(shape=(h, w), dtype=np.uint8)
    new_image *= color
    return new_image

def map_label_img(label_image):
    label_image[label_image == 0] = 255
    label_image[label_image == 1] = 30
    label_image[label_image == 2] = 0
    label_image[label_image == 3] = 150

    label_image[0, 0] = 0
    label_image[0, 1] = 30
    label_image[0, 2] = 150
    label_image[0, 3] = 255

    return label_image


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

    output_file = 'color_statistic.json'
    if len(options.output_directory) > 0:
        output_file = os.path.join(options.output_directory, output_file)
    with open(output_file) as f:
        color_count_grouped = json.load(f)

    for f in training_files:
        print("Loading file: ", f)
        image, label_image = read_complete_file(training_folder, f)
        label_image = encode_label_image(label_image)

        segmented = np.zeros(shape=label_image.shape, dtype=np.uint8);

        # Divide it into n*m chunks:
        (h, w) = image.shape[:2]
        chunk_size = 2
        for i in range(0, h, chunk_size):
            for j in range(0, w, chunk_size):
                to_i = min(i + chunk_size, h - 1)
                to_j = min(j + chunk_size, w - 1)

                # Chunk starting at i,j
                segmented[i:to_i, j:to_j] = make_color(image[i:to_i, j:to_j], color_count_grouped, options)

                unique, counts = np.unique(
                    label_image[i:to_i, j:to_j], return_counts=True)
                most_often = unique[np.argmax(counts)]
                label_image[i:to_i, j:to_j] = most_often

        correct = np.sum(label_image == segmented)
        acc = correct / float(image.shape[0] * image.shape[1])
        print(acc)


        fig = plt.figure()
        fig1 = fig.add_subplot(1, 3, 1)
        fig1.imshow(image)
        fig1 = fig.add_subplot(1, 3, 2)
        fig1.imshow(map_label_img(label_image), cmap=plt.get_cmap("Set1"))
        fig1 = fig.add_subplot(1, 3, 3)
        fig1.imshow(map_label_img(segmented), cmap=plt.get_cmap("Set1"))
        plt.show()




if __name__ == '__main__':
    main()
