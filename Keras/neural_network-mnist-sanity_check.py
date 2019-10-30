#%%
"""
Script to check functionality of binary classification algorithm.

Runs neural network on MNIST number optical character recognition benchmark.
Uses only the 0 and 1 digits to emulate binary recognition problem.
Can be used to check that neural network and custom metrics get reasonable
results.
There is currently the problem that the functions metrics_K() and
binary_confusion_matrix_K() do not calculate correct results.
Nevertheless, they are still included in this release to give the opportunity
to fix these metrics wich only use the keras.backend functions and can since
be called during model.fit() training process.

Uses Tensorflow 2.0 Release Candidate with integrated Keras Framework
Install by executing 'pip install --user tensorflow==2.0.0-rc0'
For more information visit https://www.tensorflow.org/beta/
"""
import pickle
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import Callback
import keras.backend as K
from keras.datasets import mnist
from sklearn.utils import shuffle


# Set this to the directory of the local git repository.
root_dir = './'
# Control numerical precision, can be 'float16', 'float32', or 'float64'
floatx_wanted = 'float32'
K.set_floatx(floatx_wanted)



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
                        epochs=100,
                        validation_data=[x_val, y_val],
                        verbose=1,
                        class_weight={0:1.0, 1:1.0}
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
    Raises:
        AssertionError: If the two input vectors are not of equal length, type
                        or are not numpy arrays.
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
    """Replaces all non-zero values in a list of integer values with '1'

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

    (x_train, y_train), (x_val, y_val) = mnist.load_data()

    # Use only the 0 or 1 entries
    condition_train = np.logical_or(y_train == 0, y_train == 1)
    print(condition_train)
    x_train = np.compress(condition_train, x_train, axis=0)
    y_train = np.compress(condition_train, y_train, axis=0)
    print(y_train)
    condition_val = np.logical_or(y_val == 0, y_val == 1)
    print(condition_val)
    x_val = np.compress(condition_val, x_val, axis=0)
    y_val = np.compress(condition_val, y_val, axis=0)
    print(y_val)

    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2])
    x_val = x_val.reshape(x_val.shape[0], x_val.shape[1] * x_val.shape[2])

    history, model = short_model(x_train, y_train, x_val, y_val)

    # Save history dictionary and model
    fname_history = root_dir + 'Data/NNResults/MNIST/' + 'binary' + '_history.pickle'
    with open(fname_history, 'wb') as f:
        pickle.dump(history.history, f, pickle.DEFAULT_PROTOCOL)

    fname_model = root_dir + 'Data/NNResults/MNIST/' + 'binary' + '_model.h5'
    model.save(fname_model)


# Do not execute main() when imported as module
if __name__ == '__main__':
    main(root_dir)