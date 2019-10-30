# Create the folders and copy the data (PNG images)
# for the training, validation and test data sets
# Some code was taken from 'Deep Learning with Python'
# by Francois Chollet, Listing 5.4 on page 132

import os
# Save and restore Python object structures, e.g. lists
import pickle
# Import CSV files from other tools like Synopsis Design Compiler
import csv
import numpy as np
from keras.preprocessing.image import ImageDataGenerator

# ************************* Config *************************************
# Copy file created in Klayout/klayout_scripts
fname_png_pickle = '.Data/map9v3_png.pickle'

base_dir = './map9v3'
# ************************* Config *************************************


os.mkdir(base_dir)

# Directory with data for training the neural network
train_dir = os.path.join(base_dir, 'train')
os.mkdir(train_dir)
# Directory with data for validating the results of training
validation_dir = os.path.join(base_dir, 'validation')
os.mkdir(validation_dir)
# Directory with data for testing the final performance
test_dir = os.path.join(base_dir, 'test')
os.mkdir(test_dir)

# Restore the list of filenames of PNG images created by
# Klayout/klayoutscripts
with open(fname_png_pickle, 'rb') as f:
    # 2D list with dimension len(fnames_layer)*(n_i*n_j)
    fnames_png_layers = pickle.load(f)

# TODO Take location of boxes relative to each other into account

# Number of boxes per layer
n_boxes = len(fnames_png_layers[0][:])

# Import features, like routing density, determined by commercial layout
# tools like Synopsis Design Compiler from csv files
fname_csv = 'features.csv'
features = []
# Position of wanted value in CSV row
pos = 0
# Instruct the reader to convert all non-quoted fields to type float.
csv.QUOTE_NONNUMERIC = True
with open(fname_csv, newline=' ') as f:
    ftreader = csv.reader(f, delimiter=',', quotechar='"')
    for row in ftreader:
        features.append(row[pos])

# Check if we have as many PNG boxes from Klayout
# as corresponding features
n_features = len(features)
if (n_boxes != n_features):
    print('The number of PNG images created by Klayout', n_boxes,
          'is different from the length of the feature vector', n_features)
    # TODO Raise error

features = np.asarray(features)  # Keras needs numpy arrays

"""Read the PNG images from the directories."""
# TODO Check rescale factor
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen
