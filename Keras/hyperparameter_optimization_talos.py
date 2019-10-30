# Draft to use Talos for automated optimization of hyper parameters.
# By Dr. Robert Fischbach


import pickle
import talos as ta  # version: git @daily-dev -> 0.6.3
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from sklearn.utils import shuffle
import tensorflow as tf

benchmark_names=['mgc_des_perf_1', 'mgc_des_perf_a', 'mgc_des_perf_b',
                 'mgc_fft_1', 'mgc_fft_2', 'mgc_fft_a',
                 'mgc_matrix_mult_1', 'mgc_matrix_mult_a', 'mgc_pci_bridge32_a',
                 'mgc_pci_bridge32_b']
benchmark_name = benchmark_names[3]
folder_prefix_training = './Data/Training_Data_and_Labels/design_placed_training_data_'
folder_prefix_labels = './Data/Training_Data_and_Labels/design_routed_labels_shorts_'


fh = open(folder_prefix_training + benchmark_name + '.pickle', 'br')
features = pickle.load(fh)
fh.close()

fh = open(folder_prefix_labels + benchmark_name + '.pickle', 'br')
labels = pickle.load(fh)
fh.close()

assert(len(features) == len(labels))

x, y = shuffle(features, labels, random_state=666)
x = np.array(x)
x = x.reshape(x.shape[0], x.shape[1] * x.shape[2])
print('x.shape:', x.shape)
y = np.array(y)

#%%

train_to_validation_ratio = 0.8
assert(0 < train_to_validation_ratio < 1)

num_training_samples = int(len(x) * train_to_validation_ratio)
x_train = x[:num_training_samples]
y_train = y[:num_training_samples]
x_val = x[num_training_samples:]
y_val = y[num_training_samples:]


#%%
# Talos parameters
p = {'activation': ['relu', 'elu', 'softmax', 'linear'],
     'optimizer': ['Nadam', 'Adam', 'SGD', 'RMSprop', 'Adadelta'],
     'losses': ['binary_crossentropy'],
     'hidden_layers': [1, 2, 3],
     'shapes': ['brick'],  # <<< required
     'first_neuron': [8, 16, 32, 64, 128],  # <<< required
     'dropout': [.1, .2, .3, .4],  # <<< required
     # 'batch_size': [30, 100, 150, 200, 500],
     'epochs': [20, 50, 100 ,200]}


def short_model(x_train, y_train, x_val, y_val, params):

    model = Sequential()
    model.add(Dense(128, input_dim=x_train.shape[1], activation=params['activation']))
    ta.utils.hidden_layers(model, params, 1)
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=params['optimizer'], loss=params['losses'], metrics=['accuracy'])

    out = model.fit(x_train, y_train,
                    batch_size=params['batch_size'],
                    epochs=params['epochs'],
                    validation_data=[x_val, y_val],
                    verbose=0,
                    class_weight={0:1.0, 1:25.0}
                    )

    return out, model


scan_object = ta.Scan(x_train, y_train, x_val=x_val, y_val=y_val, model=short_model,random_method='quantum', reduction_method='correlation', params=p, experiment_name='2019-08-26')