#%%
"""
Runs neural network on data in folder 'Data/Training_Data_and_labels'
In contrast to the equivalent scripts in folder
JupyterNotebooksForGoogleColab these are meant to be run directly on a
local computer.

Adapted from 'main.py' on 2019-08-14 by Robert Fischbach.
Uses parameters guessed manually, without using Talos.
"""
import pickle
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import Callback
import keras.backend as K
from sklearn.utils import shuffle

# ************************* Config *************************************
# Change this to the directory of the local git repository.
root_dir = './'
# Control numerical precision, can be 'float16', 'float32', or 'float64'
floatx_wanted = 'float32'
K.set_floatx(floatx_wanted)
# Choose wether to normalize data to the mean and standard deviation of the
# training data. Not clear if it improves or hampers performance.
normalize_data = True
# Use a unique name for each experiment so that the results are saved separately
experiment_name = '1500epochs'
# ************************* Config *************************************


def short_model(x_train, y_train, x_val, y_val):
    """Run a Keras sequential Neural Network.

    Args:
        x_train (numpy array):                  Training features
        y_train (numpy array):                  Training labels
        x_val (numpy array):                    Validation features
        y_val (numpy array):                    Validation labels
        metrics_cb (inst of Keras Callback()):  Callback metric

    Returns:
        history [keras.callbacks.History]: Data about training history
        model   [keras.models.Sequential]: Fitted Neural Network
    """
    model = Sequential()
    model.add(Dense(16, input_shape=(x_train.shape[1],), activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='Nadam', loss='binary_crossentropy',
                  metrics=['acc', accuracy_K, specificity,
                           true_positive_rate, false_positive_rate,
                           matthews_correlation_coefficient,
                           num_total, num_shorts, TP, FN, TN, FP
                           ])

    history = model.fit(x_train, y_train,
                        batch_size= 10000,   #int(y_train.shape[0] * 0.1),
                        epochs=500,
                        validation_data=[x_val, y_val],
                        verbose=1,
                        class_weight={0:1.0, 1:25.0}
                        )
    return history, model


def convert_to_bools(nparray):
    """Replaces all above-one values in a list of integer values with '1'

    Args:
        nparray:   Numpy array

    Returns:
        binary_nparray:    Numpy array consisting of '0's and '1's
    """
    ones = np.ones(np.shape(nparray))
    binary_nparray = np.minimum(nparray, ones)
    return binary_nparray


def main(root_dir):
    """Run neural networks with imported feature and label data and save them.

    Args:
        root_dir (string):      directory with the Data/NNResults and
                                Data/Training_Data_and_Labels folders

    """

    # Leave out  mgc_fft_2, like Tabrizi 2018, to have a 'clean' test case.
    benchmark_names = ['mgc_des_perf_1', 'mgc_des_perf_a', 'mgc_des_perf_b',
                    'mgc_fft_1', 'mgc_fft_a',
                    'mgc_matrix_mult_1', 'mgc_matrix_mult_a',
                    'mgc_pci_bridge32_a', 'mgc_pci_bridge32_b']
    features = np.zeros((0, 9, 8))
    labels = np.zeros((0,))
    for i in range(len(benchmark_names)):
        benchmark_name = benchmark_names[i]
        folder_prefix_training = root_dir + 'Data/Training_Data_and_Labels/design_placed_training_data_'
        folder_prefix_labels = root_dir + 'Data/Training_Data_and_Labels/design_routed_labels_shorts_'

        train_to_validation_ratio = 0.8
        assert(0 < train_to_validation_ratio < 1)

        fh = open(folder_prefix_training + benchmark_name + '.pickle', 'br')
        features_cur = pickle.load(fh)
        fh.close()

        fh = open(folder_prefix_labels + benchmark_name + '.pickle', 'br')
        labels_cur = pickle.load(fh)
        fh.close()

        labels_cur = np.asarray(labels_cur)
        features_cur = np.asarray(features_cur)

        # Convert numbers of shorts to binary 0s and 1s (exist or not).
        labels_cur = convert_to_bools(labels_cur)
        print(labels_cur)
        labels_cur = np.asarray(labels_cur)

        print('labels.shape', labels.shape)
        print('labels_cur.shape', labels_cur.shape)
        print('features_cur.shape', features_cur.shape)
        print('features.shape', features.shape)
        labels = np.concatenate((labels, labels_cur), axis=0)
        features = np.concatenate((features, features_cur), axis=0)
    assert(len(features) == len(labels))
    print('labels.shape:', labels.shape)
    print('features.shape:', features.shape)
    benchmark_name = 'combined_without_mgc_fft_2'

    print('\nFitting model for benchmark', benchmark_name)

    x, y = shuffle(features, labels, random_state=666)
    x = np.array(x)

    x = x.reshape(x.shape[0], x.shape[1] * x.shape[2])
    y = np.array(y)
    print('Total number of shorts:', np.sum(y))
    print('x.shape:', x.shape)
    print('y.shape:', y.shape)

    num_training_samples = int(len(x) * train_to_validation_ratio)
    x_train = x[:num_training_samples]
    y_train = y[:num_training_samples]
    x_val = x[num_training_samples:]
    y_val = y[num_training_samples:]

    if normalize_data:
        # Normalize Data as in 'Deep Learning with Python', p.86
        mean = x_train.mean(axis=0)
        x_train -= mean
        std = x_train.std(axis=0)
        x_train /= std
        # Normalize x_val with mean and std from x_train to prevent data leakage
        x_val -= mean
        x_val /= std

    history, model = short_model(x_train, y_train, x_val, y_val)

    # Save history dictionary and model
    fname_history = root_dir + 'Data/NNResults/' + benchmark_name + '_' + \
                    experiment_name + '_history.pickle'
    with open(fname_history, 'wb') as f:
        pickle.dump(history.history, f, pickle.DEFAULT_PROTOCOL)

    fname_model = root_dir + 'Data/NNResults/' + benchmark_name + '_' + \
                  experiment_name + '_model.h5'
    model.save(fname_model)


# Do not execute main() when imported as module
if __name__ == '__main__':
    main(root_dir)
