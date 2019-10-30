"""Displays the metrics of Keras History dictionaries in folder 'Data/NNResults'.
"""
import pickle
import matplotlib.pyplot as plt



# ************************* Config *************************************
# Choose wether custom metrics like TPR or MCC were calculated during training.
custom_metrics = False
# Set this to the directory of the local git repository.
root_dir = './'
combine_data = True
benchmark_names_comb = ['combined_without_mgc_fft_2_1500epochs_normalized',
                   'combined_without_mgc_fft_2_3000epochs_normalized',
                    'combined_without_mgc_fft_2_3000epochs',
                   'combined_without_mgc_fft_2_6000epochs_normalized',
                   'combined_without_mgc_fft_2_750epochs_normalized',
                   'combined_without_mgc_fft_2_600epochs_normalized',
                   'combined_without_mgc_fft_2_6000epochs',
                    ]
# ************************* Config *************************************

def print_history(fname_history, custom_metrics):
    with open(fname_history, 'rb') as f:
        history_dict = pickle.load(f)
    print('Metrics for benchmark', benchmark_name + ' for last epoch:')
    if custom_metrics:
    # Extract metrics from last epoch of training
        SPC = history_dict['specificity'][-1]
        ACC = history_dict['accuracy_K'][-1]
        accuracy = history_dict['acc'][-1]
        TPR = history_dict['true_positive_rate'][-1]
        FPR = history_dict['false_positive_rate'][-1]
        MCC = history_dict['matthews_correlation_coefficient'][-1]
        TP = history_dict['TP'][-1]
        TN = history_dict['TN'][-1]
        FP = history_dict['FP'][-1]
        FN = history_dict['FN'][-1]
        num_total = history_dict['num_total'][-1]
        num_shorts = history_dict['num_shorts'][-1]

        print('Number of samples:', num_total)
        print('Number of shorts:', num_shorts)
        print('Number of True Positives:', TP)
        print('Number of True Negatives:', TN)
        print('Number of False Positives:', FP)
        print('Number of False Negatives:', FN)
        print('accuracy:', accuracy)
        print('Sensitivity or True Positive Rate (TPR):', TPR)
        print('Specificity (SPC):', SPC)
        print('False Alarm Rate (FPR):', FPR)
        print('Accuracy (ACC):', ACC)
        print('Matthews Correlation Coefficient (MCC):', MCC)
    else:
        entries = ['loss', 'acc', 'binary_crossentropy', 'val_loss', 'val_acc', 'val_binary_crossentropy']
        for entry in entries:
            print(entry + ':', history_dict[entry][-1])
            if (entry == 'loss') or (entry == 'val_loss'):
                plt.semilogy(range(1, len(history_dict[entry][:]) + 1), history_dict[entry][:])
            else:
                plt.plot(range(1, len(history_dict[entry][:]) + 1), history_dict[entry][:])
            plt.xlabel('Epochs')
            plt.ylabel(entry)
            fname_fig = root_dir + 'Data/NNResults/figures/'+ benchmark_name + '_' + entry + '.png'
            plt.savefig(fname_fig)
            plt.clf()
    print('')



if combine_data:
    for benchmark_name in benchmark_names_comb:
        fname_history = root_dir + 'Data/NNResults/' + benchmark_name + '_history.pickle'
        print_history(fname_history, custom_metrics)

else:
    benchmark_names=['mgc_des_perf_1', 'mgc_des_perf_a', 'mgc_des_perf_b',
                    'mgc_fft_1', 'mgc_fft_2', 'mgc_fft_a',
                    'mgc_matrix_mult_1', 'mgc_matrix_mult_a', 'mgc_pci_bridge32_a',
                    'mgc_pci_bridge32_b']
    for benchmark_name in benchmark_names:
        # Load history dictionary from disk
        fname_history = root_dir + 'Data/NNResults/' + benchmark_name + '_history.pickle'

