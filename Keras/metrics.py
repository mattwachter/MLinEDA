import pickle
import numpy as np
import tensorflow.keras.models as models
from sklearn.utils import shuffle
import csv

# ************************* Config *************************************
# Change this to the directory of the local git repository.
root_dir = './'
# Controls numerical precision, can be 'float16', 'float32', or 'float64'
# Set to the datatype used in the loaded Keras model.
floatx_cur = 'float32'
# Choose wether to use only the validation data of the benchmarks or all data.
val_data = True
# Choose whether to escape '_' to '\_' characters in CSV output.
# Useful for import to LaTex with package 'csvsimple'
output_for_latex = True
# Choose wether to normalize data to the mean and standard deviation of the
# training data. Not clear if it improves or hampers performance.
normalize_data = True
# Calculate and save metrics with these models
benchmark_names_comb = [  'combined_without_mgc_fft_2_1500epochs',
                    'combined_without_mgc_fft_2_1500epochs_normalized',
                    'combined_without_mgc_fft_2_1000epochs_normalized',
                   'combined_without_mgc_fft_2_3000epochs_normalized',
                    'combined_without_mgc_fft_2_3000epochs',
                   'combined_without_mgc_fft_2_6000epochs_normalized',
                   'combined_without_mgc_fft_2_750epochs_normalized',
                   'combined_without_mgc_fft_2_600epochs_normalized',
                   'combined_without_mgc_fft_2_6000epochs',
                   'combined_without_mgc_fft_2_1500epochs',
                    ]

# ************************* Config *************************************


def metrics_numpy(y_true, y_pred):
    """Calculates quality metrics from numpy arrays of true and predicted values.

    Definitions from paper 'A Machine Learning Framework to Identify Detailed
    Routing Short Violations from a Placed Netlist' by Tabrizi et al. 2018

    Args:
        y_true (numpy array): Array of 'true' 0s and 1s
        y_pred (numpy array): Array of predicted 0s and 1s

    Returns:
        TN (float):     Number of true negative samples
        FP (float):     Number of false positive samples
        FN (float):     Number of false negative samples
        TP (float):     Number of true positive samples
        TPR (float):    True Positive Rate ([0:1])
        SPC (float):    Specificity ([0:1])
        FPR (float):    False Positive Rate ([0:1])
        ACC (float):    Accuracy ([0:1])
        MCC (float):    Matthews Correlation Coefficient, ([-1:+1]), +1 is best
    Raises:
        AssertionError: If the two input vectors are not of equal length, type
                        or are not numpy arrays.
    """
    assert(type(y_true) == type(y_pred))
    assert(type(y_true) == np.ndarray)
    assert(len(y_true) == len(y_pred))

    # y_true and y_pred need to have exactly the same shape.
    # We assume that the only difference is between (len,) and (len, 1)
    y_true = y_true.reshape(len(y_true), 1)
    y_pred = y_pred.reshape(len(y_pred), 1)


    TN, FP, FN, TP = binary_confusion_matrix_numpy(y_true, y_pred)

    # Sensitivity or True Positive Rate (TPR)
    TPR = TP/(TP + FN)
    # Specificity (SPC)
    SPC = TN/(TN+FP)
    # False Alarm (FPR)
    FPR = FP/(TN + FP)
    #  Accuracy (ACC)
    ACC = (TP + TN)/(TP + TN + FP + FN)
    # Matthews Correlation Coefficient (MCC); Values from -1 to +1, +1 is best.
    MCC = (TP*TN - FP*FN)/np.sqrt((TP + FP)*(TP + FN)*(TN + FP)*(TN + FN))
    return TP, TN, FP, FN, TPR, SPC, FPR, ACC, MCC


def binary_confusion_matrix_numpy(y_true, y_pred):
    """Numpy implementation of a binary confusion matrix.

    Should be independent of frameworks and work with tensors.

    Args:
        y_true (numpy array, float): Array of 'true' 0s and 1s
        y_pred (numpy array, float): Array of predicted 0s and 1s

    Returns:
        TN (float):     Number of true negative samples
        FP (float):     Number of false positive samples
        FN (float):     Number of false negative samples
        TP (float):     Number of true positive samples

    Raises:
        ValueError:     Values need to be either 0 or 1
    """
    # Test if all array elements are either 0.0 or 1.0
    binary_bool = np.all(np.logical_or(y_true == 0.0, y_true == 1.0))
    if not binary_bool:
        raise ValueError('Values of y_true need to be either 0 or 1')
    binary_bool = np.all(np.logical_or(y_pred == 0.0, y_pred == 1.0))
    if (not binary_bool):
        raise ValueError('Values of y_pred need to be either 0 or 1')

    TN = np.sum(np.logical_and(y_true == 0.0, y_pred == 0.0)).astype(floatx_cur)
    FP = np.sum(np.logical_and(y_true == 0.0, y_pred == 1.0)).astype(floatx_cur)
    FN = np.sum(np.logical_and(y_true == 1.0, y_pred == 0.0)).astype(floatx_cur)
    TP = np.sum(np.logical_and(y_true == 1.0, y_pred == 1.0)).astype(floatx_cur)

    return TN, FP, FN, TP


def convert_to_bools(nparray):
    """Replaces all values below 0.5 with 0.0 and the rest with 1.0'

    Args:
        nparray:   Numpy array

    Returns:
        binary_nparray:    Numpy array consisting of '0's and '1's
    """
    ones = np.ones(np.shape(nparray))
    binary_nparray = np.minimum(np.round(nparray), ones)
    return binary_nparray


def append_csv(fname_csv, line_list):

    for i in [0]:
        if output_for_latex:
            # Escape problematic characters for inclusion in Latex
            print(line_list[i])
            item_list = line_list[i].split('_')
            line_list[i] = '\_'.join(item_list)
            print(line_list[i])
        else:
            pass
    for i in range(1, 3):
        # Format number to int
        line_list[i] = "{:n}".format(line_list[i])
    for i in range(3, 8):
        # Format number to int
        line_list[i] = "{:.3f}".format(line_list[i])
    with open(fname_csv, 'a', newline='') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',')
        csvwriter.writerow(line_list)

def begin_csv(fname_csv, line_list):
    with open(fname_csv, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',')
        csvwriter.writerow(line_list)


for benchmark_name_training in benchmark_names_comb:
    print('\n\nCalculating metrics based on predictions by model', benchmark_name_training)
    fname_model = root_dir + 'Data/NNResults/' + benchmark_name_training +'_model.h5'
    model_cur = models.load_model(fname_model, compile=False)

    fname_csv = root_dir + 'Data/NNResults/metrics_' + benchmark_name_training + '.csv'
    # Write column headings
    line_list = ['Benchmark', 'n', 'TP',
                'TPR', 'SPC', 'FPR', 'ACC', 'MCC']
    begin_csv(fname_csv, line_list)

    benchmark_names=['mgc_des_perf_1', 'mgc_des_perf_a', 'mgc_des_perf_b',
                    'mgc_fft_1', 'mgc_fft_2', 'mgc_fft_a',
                    'mgc_matrix_mult_1', 'mgc_matrix_mult_a', 'mgc_pci_bridge32_a',
                    'mgc_pci_bridge32_b']
    for benchmark_name_test in benchmark_names:
        folder_prefix_training = root_dir + 'Data/Training_Data_and_Labels/design_placed_training_data_'
        folder_prefix_labels = root_dir + 'Data/Training_Data_and_Labels/design_routed_labels_shorts_'
        train_to_validation_ratio = 0.8
        assert(0 < train_to_validation_ratio < 1)

        fh = open(folder_prefix_training + benchmark_name_test + '.pickle', 'br')
        features = pickle.load(fh)
        fh.close()

        fh = open(folder_prefix_labels + benchmark_name_test + '.pickle', 'br')
        labels = pickle.load(fh)
        fh.close()

        assert(len(features) == len(labels))

        if val_data:
            x, y = shuffle(features, labels, random_state=666)
            x = np.array(x, dtype=floatx_cur)
            x = x.reshape(x.shape[0], x.shape[1] * x.shape[2])
            y = np.array(y, dtype=floatx_cur)

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

            y_val_pred = model_cur.predict(x_val)

            # For proper comparison the samples need to consist only of 0s and 1s
            y_val = convert_to_bools(y_val)
            y_val_pred = convert_to_bools(y_val_pred)

            num_total = y_val.size
            num_shorts = np.sum(y_val)

            TP, TN, FP, FN, TPR, SPC, FPR, ACC, MCC = metrics_numpy(y_val, y_val_pred)

            print('Metrics for validation data of benchmark', benchmark_name_test,
                'predicted by a model trained on data from', benchmark_name_training + ':')
        else:
            x = features
            y = labels
            x = np.array(x, dtype=floatx_cur)
            x = x.reshape(x.shape[0], x.shape[1] * x.shape[2])
            y = np.array(y, dtype=floatx_cur)


            if normalize_data:
                # Normalize Data as in 'Deep Learning with Python', p.86
                mean = x.mean(axis=0)
                x -= mean
                std = x.std(axis=0)
                x /= std

            y_true = y
            y_pred = model_cur.predict(x)

            # For proper comparison the samples need to consist only of 0s and 1s
            y_true = convert_to_bools(y_true)
            y_pred = convert_to_bools(y_pred)

            num_total = y_true.size
            num_shorts = np.sum(y_true)

            TP, TN, FP, FN, TPR, SPC, FPR, ACC, MCC = metrics_numpy(y_true, y_pred)
            print('Metrics for data of benchmark', benchmark_name_test,
                'predicted by a model trained on data from', benchmark_name_training + ':')
        line_list = [benchmark_name_test, num_total, num_shorts,
                    TPR, SPC, FPR, ACC, MCC]
        # line_list = [benchmark_name_test, num_total, num_shorts,
        #              TP, TN, FP, FN, TPR, SPC, FPR, ACC, MCC]
        append_csv(fname_csv, line_list)
        print('Number of samples:', num_total)
        print('Number of shorts:', num_shorts)
        print('Number of True Positives:', TP)
        print('Number of True Negatives:', TN)
        print('Number of False Positives:', FP)
        print('Number of False Negatives:', FN)
        print('Sensitivity or True Positive Rate (TPR):', TPR)
        print('Specificity (SPC):', SPC)
        print('False Alarm Rate (FPR):', FPR)
        print('Accuracy (ACC):', ACC)
        print('Matthews Correlation Coefficient (MCC):', MCC)
        print('')
    print('Wrote data to CSV file', fname_csv)