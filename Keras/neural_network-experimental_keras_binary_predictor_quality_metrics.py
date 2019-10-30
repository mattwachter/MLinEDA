#%%
"""
Runs neural network on data in folder 'Data/Training_Data_and_labels'

Experimental custom metrics to be calculated and displayed during
training are included in this version of neural_networks.py
In contrast to the functions used in metrics.py (which return correct
results) these use just the keras.backend (K) functions to handle
Tensors without eager execution. This enables the calculation and
display of the custom metrics during training.
Unfortunately these do not yet return meaningful results; e.g. the
number of shorts is not even close to an integer value.

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


def metrics_K(y_true, y_pred):
    """Calculates quality metrics from true and predicted Tensorflow values.

    Uses just the keras.backend (K) functions to handle Tensors without eager
    execution.
    Definitions from paper 'A Machine Learning Framework to Identify Detailed
    Routing Short Violations from a Placed Netlist' by Tabrizi et al. 2018

    Args:
        y_true (Tensor): Array of 'true' 0s and 1s
        y_pred (Tensor): Array of predicted 0s and 1s

    Returns:
        TN (Tensor):     Number of true negative samples
        FP (Tensor):     Number of false positive samples
        FN (Tensor):     Number of false negative samples
        TP (Tensor):     Number of true positive samples
        TPR (Tensor):    True Positive Rate ([0:1])
        SPC (Tensor):    Specificity ([0:1])
        FPR (Tensor):    False Positive Rate ([0:1])
        ACC (Tensor):    Accuracy ([0:1])
        MCC (Tensor):    Matthews Correlation Coefficient, ([-1:+1]), +1 is best
    """
    # Get predicted value by rounding the probability value to the nearest int.
    y_pred = K.round(y_pred)

    # assert(K.int_shape(y_true) == K.int_shape(y_pred))
    # Total number of elements in y_true or y_pred
    num_total = K.prod(K.shape(y_true))
    # Cast from int32 to floatx_wanted to ensure compatibility with other values
    num_total = K.cast(num_total, floatx_wanted)

    # Dot product gives sum of instances where y_true and y_pred are both 1.
    TP = K.dot(y_true, K.transpose(y_pred))
    FN = K.sum(y_true) - TP
    falses = K.not_equal(y_true, y_pred)
    # Convert from bools to 0. and 1.
    falses = K.cast(falses, floatx_wanted)
    FP = K.dot(y_pred, K.transpose(falses))
    TN = num_total - FP

    # Use K.epsilon to avoid division by zero.
    eps = K.epsilon()

    # Sensitivity or True Positive Rate (TPR)
    TPR = TP/(TP + FN + eps)
    # Specificity (SPC)
    SPC = TN/(TN + FP + eps)
    # False Alarm (FPR)
    FPR = FP/(TN + FP + eps)
    #  Accuracy (ACC)
    ACC = (TP + TN)/(TP + TN + FP + FN + eps)
    # Matthews Correlation Coefficient (MCC); Values from -1 to +1, +1 is best.
    MCC = (TP*TN - FP*FN)/K.sqrt((TP + FP + eps) * (TP + FN + eps) *
                                 (TN + FP + eps) * (TN + FN + eps))
    return TP, TN, FP, FN, TPR, SPC, FPR, ACC, MCC


def num_shorts(y_true, y_pred):
    return K.sum(y_true)


def num_total(y_true, y_pred):
    return K.prod(K.shape(y_true))


def TP(y_true, y_pred):
    TP, TN, FP, FN, TPR, SPC, FPR, ACC, MCC = metrics_K(y_true, y_pred)
    return TP


def TN(y_true, y_pred):
    TP, TN, FP, FN, TPR, SPC, FPR, ACC, MCC = metrics_K(y_true, y_pred)
    return TN


def FP(y_true, y_pred):
    TP, TN, FP, FN, TPR, SPC, FPR, ACC, MCC = metrics_K(y_true, y_pred)
    return FP


def FN(y_true, y_pred):
    TP, TN, FP, FN, TPR, SPC, FPR, ACC, MCC = metrics_K(y_true, y_pred)
    return FN


def true_positive_rate(y_true, y_pred):
    """Returns True Positive Rate.

    Wrapper function of metrics_K()

    Args:
        y_true (Tensorflow Tensor): Array of 'true' 0s and 1s
        y_pred (Tensorflow Tensor): Array of predicted 0s and 1s

    Returns:
        specificity_loss (Tensor): True Positive Rate ([0:1])
    """
    TP, TN, FP, FN, TPR, SPC, FPR, ACC, MCC = metrics_K(y_true, y_pred)
    return TPR


def specificity(y_true, y_pred):
    """Returns specificity of true and predicted values.

    Wrapper function of metrics_K()

    Args:
        y_true (Tensorflow Tensor): Array of 'true' 0s and 1s
        y_pred (Tensorflow Tensor): Array of predicted 0s and 1s

    Returns:
         (Tensor): Specificity ([0:1])
    """
    TP, TN, FP, FN, TPR, SPC, FPR, ACC, MCC = metrics_K(y_true, y_pred)
    return SPC


def false_positive_rate(y_true, y_pred):
    """Returns False Positive Rate of true and predicted values.

    Wrapper function of metrics_K()

    Args:
        y_true (Tensorflow Tensor): Array of 'true' 0s and 1s
        y_pred (Tensorflow Tensor): Array of predicted 0s and 1s

    Returns:
        (Tensor): False Positive Rate  ([0:1])
    """
    TP, TN, FP, FN, TPR, SPC, FPR, ACC, MCC = metrics_K(y_true, y_pred)
    return FPR


def accuracy_K(y_true, y_pred):
    """Returns accuracy of true and predicted values.

    Wrapper function of metrics_K()
    ACC = (TP + TN)/(TP + TN + FP + FN)

    Args:
        y_true (Tensorflow Tensor): Array of 'true' 0s and 1s
        y_pred (Tensorflow Tensor): Array of predicted 0s and 1s

    Returns:
        (Tensor): Accuracy ([0:1])
    """
    TP, TN, FP, FN, TPR, SPC, FPR, ACC, MCC = metrics_K(y_true, y_pred)
    return ACC


def matthews_correlation_coefficient(y_true, y_pred):
    """Returns Matthews Correlation Coefficient of true and predicted values.

    Wrapper function of metrics_K()
    MCC = (TP*TN-FP*FN)/sqrt((TP + FP)*(TP+FN)*(TN+FP)*(TN+FN))
    Args:
        y_true (Tensorflow Tensor): Array of 'true' 0s and 1s
        y_pred (Tensorflow Tensor): Array of predicted 0s and 1s

    Returns:
        (Tensor): Accuracy ([0:1])
    """
    TP, TN, FP, FN, TPR, SPC, FPR, ACC, MCC = metrics_K(y_true, y_pred)
    return SPC


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
