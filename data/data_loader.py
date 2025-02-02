from sklearn.preprocessing import StandardScaler
import os
import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
import yaml


# Set random seed.
with open(r'config\config.yaml') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
SEED = config['SEED']
random.seed(SEED)


def load_data(path:str, file_name:str, partition=False):
    """
    Load dataset.

    Arge:
        path: path of directory that contaning dataset.
        file_name: csv file that contaning image names and labels.
        partition: for work on a partiotion of data (1000 samples).

    Returns:
        x_data: images
        y_data: labels
    """

    x_data = []
    y_data = []
    with open(os.path.join(path, file_name), 'r') as csvfile:
        datareader = csv.reader(csvfile, delimiter=',')
        datareader.__next__() # don't use first row (header of file) as image name and label

        if partition:
            datareader = list(datareader)[:1000]

        for row in datareader:
            im = cv2.imread(os.path.join(path, row[0]), cv2.IMREAD_UNCHANGED)
            x_data.append(im)
            y_data.append(row[1])

    return x_data, np.array(y_data, dtype=np.int8)


def show_images(X, y):
    """
    Display 50 random samples of images.
    
    Arges:
        X: input images
        y: labels
    """

    fig, axs = plt.subplots(5, 10)
    nums = random.sample(range(len(y)), 50)
    n = 0
    for i in range(5):
        for j in range(10):
            axs[i, j].imshow(X[nums[n]], cmap="gray")
            axs[i, j].title.set_text(f"lbl:{y[nums[n]]}")
            axs[i, j].set_yticklabels([]) # off y axis labels
            axs[i, j].set_xticklabels([]) # off x axis labels
            n += 1
    plt.subplots_adjust(hspace=1) # set the spacing between subplots
    plt.show()


def histogram(y):
    """
    Compute histogram of label frequencies.

    Args:
        y: labels
    """

    unique, counts = np.unique(y, return_counts=True)

    plt.bar(unique, counts)
    plt.show()


def resizing(X)->np.array:
    """
    Resize input images to desired size.

    Args:
        X: input images

    Returns:
        x_re: resized images
    """

    x_re = np.zeros((len(X), 28, 28), dtype=np.float32)
    for i in range(len(X)):
        x_re[i] = cv2.resize(X[i], dsize=(28, 28), interpolation=cv2.INTER_LINEAR)
    
    return x_re


def flatten(X)->np.array:
    """
    Reshape images to 1-D array.

    Args:
        X: input images

    Rturns:
        X: flat images. shape: (number of samples, 1, 28*28)
    """

    X = X.reshape(X.shape[0], 28*28)
    X = X.astype('float32')

    return np.array(X)


def normalization(X:np.array)->np.array:
    """
    Normalize data.
    X_new = (X - X_min)/(X_max - X_min)
    for images: max=255, min=0

    Args:
        X: input data

    Rturns:
        X: normalize data.
    """

    # X /= 255
    X = (X - np.mean(X)) / np.std(X)
    return np.array(X)


def to_categorical(y:np.array) -> np.array:
    """
    One-hot encoding of labels.

    Args:
        y: lables

    Returns:
        y_c: one-hot lables
    """

    num_classes = len(np.unique(y))
    y_c = np.eye(num_classes)[y]

    return np.array(y_c)
