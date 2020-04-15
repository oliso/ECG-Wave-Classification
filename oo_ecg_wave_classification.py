"""

ECG segmentation tool.

This tool was built around proprietary data from POWERFUL MEDICAL which is not
public. I am currently trying to source a different data set to run this on,
but for the time being, I am publishing the code at least.

Author: Oliver Osvald (oloosvald@gmail.com)

"""

# 0. Set up libs -------------------------------------------------------------

import os
import json
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt

from typing import Dict
import random
import scipy as sp
from tensorflow import keras


# 1. Data import & partition -------------------------------------------------

# Data format:
# {'data': {
#     '<lead_name>': {
#         'ecg': [[]],
#         'label': [[]],
#         'fs': int} },
# 'legend': {0: 'none', 1: "p_wave", 2: "qrs", 3: "t_wave", 4: "extrasystole"}}

# Import function:
def import_signal(path: str) -> Dict[str, dict]:
    """Load a single signal sample."""
    with open(path) as f:
        signal_sample = json.load(f)
        return signal_sample


# Select data set range (original data contained 290 leads):
all_lead_range = range(289)

# Shuffle to get a mix of sources
np.random.seed(123)
randomised_ecgs = np.array(all_lead_range)
random.shuffle(randomised_ecgs)

# Decided to use a 80/10/10 split for now:
train_range = randomised_ecgs[0:230]
validation_range = randomised_ecgs[231:260]
test_range = randomised_ecgs[261:289]

# Combine leads for TRAIN SET:
train_leads = []
for lead in train_range:
    data_root = 'data'
    ecg_id = lead  # choose ecg id from [0, 289]

    # load split file containing paths to the data
    df = pd.read_csv(os.path.join(data_root, 'split.csv'))

    # Load single ecg with above f'n
    path = df.iloc[ecg_id]['name']
    lead_name = path.split('/')[1][:-5]
    signal_sample = import_signal(os.path.join(data_root, path))
    train_leads.append(signal_sample)

# Combine leads for VALID SET:
valid_leads = []
for lead in validation_range:
    data_root = 'data'
    ecg_id = lead  # choose ecg id from [0, 289]

    # load split file containing paths to the data
    df = pd.read_csv(os.path.join(data_root, 'split.csv'))

    # Load single ecg with above f'n
    path = df.iloc[ecg_id]['name']
    lead_name = path.split('/')[1][:-5]
    signal_sample = import_signal(os.path.join(data_root, path))
    valid_leads.append(signal_sample)

# Combine leads for TEST SET:
test_leads = []
for lead in test_range:
    data_root = 'data'
    ecg_id = lead

    # load split file containing paths to the data
    df = pd.read_csv(os.path.join(data_root, 'split.csv'))

    # Load single ecg with above f'n
    path = df.iloc[ecg_id]['name']
    lead_name = path.split('/')[1][:-5]
    signal_sample = import_signal(os.path.join(data_root, path))
    test_leads.append(signal_sample)


# 2. Processing and loading the training set ---------------------------------

# Will be using these functions later (is there equiv in base python?)
def every2(a):
    """Skip every other item in a list."""
    return a[::2]


def every3(a):
    """Skip every second and third item in a list."""
    return a[::3]


# :
def preprocess(indices, m):
    """Pre-process pipeline function."""
    # f'n takes two input args:
    # indices - which leads to include in the new data set
    # m - number of time steps on each side of the located peak

    train_leads_indices = indices
    all_peak_index_subsets = []
    all_ecg_subsets = []
    all_peak_labels = []

    for k in train_leads_indices:

        # Pick up a lead
        signal_sample = train_leads[k]

        # Find out lead name
        for key in signal_sample['data']:
            lead_name = key

        # ecg signal as numpy array
        ecg = np.array(signal_sample['data'][lead_name]['ecg'])
        lead_labels = np.array(signal_sample['data'][lead_name]['label'])

        # Locate all peaks using a scipy.signal method:
        peaks = sp.signal.find_peaks(ecg[0])

        """
        Below is a portion of code which I used to visualise where the peaks
        are detected in the signal, to ensure that perceived peaks are really
        peaks.
        """

        # # Create an array showing signal peaks only:
        # peaks_only = (ecg[0]) * 0
        # for p in range(len(list(peaks_only))):
        #     print("p is "+str(p))
        #     for s in peaks[0]:
        #         print("s is "+str(s))
        #         if p == s:
        #             peaks_only[p] = ecg[0][p]
        #         else:
        #             continue

        # # Check peaks are captured properly:
        # plt.plot(ecg[0])
        # plt.plot(peaks_only, 'ro')
        # plt.ylabel('ECG')
        # plt.show()

        """
        My model will take input features which will utilise n time steps
        before and after the peaks detected. However, due to differing sample
        rates, I need to make sure to take inputs from equally-sized time
        segments.

        In the three datasets, the leads have fs = 112.1hz, 257hz, and 360hz.
        For simplicity (and higher consistency with the slowest sampling freq),
        I choose to use every other time step where fs = 257 and every third
        time step where fs = 360.

        That should bring the overall sampled segment size to be roughly equal
        in the three cases (112:128:120).
        """

        # Get sampling freq:
        lead_fs = signal_sample['data'][lead_name]['fs']
        peak_index_subsets = []
        ecg_subsets = []
        peak_labels = []

        n = m

        for p in peaks[0]:
            print(f"Processing train lead no. {k}, peak no.{p}")

            if lead_fs < 113:
                p_low = p - n
                p_top = p + n + 1

                try:
                    peak_index_range = list(range(p_low, p_top))

                    # Append peak detail lists:
                    peak_index_subsets.append(peak_index_range)
                    ecg_subsets.append(list(ecg[0][peak_index_range]))
                    peak_labels.append(lead_labels[0][p])

                except IndexError:
                    continue

            elif lead_fs == 257:
                # take every other (make sure to take the peak index p)
                p_low = p - n * 2
                p_top = p + (n + 1) * 2

                try:
                    peak_index_range = every2(list(range(p_low, p_top)))

                    # Append peak detail lists:
                    peak_index_subsets.append(peak_index_range)
                    ecg_subsets.append(list(ecg[0][peak_index_range]))
                    peak_labels.append(lead_labels[0][p])

                except IndexError:
                    continue

            elif lead_fs == 360:
                # take every third (make sure to take the peak index p)
                p_low = p - n * 3
                p_top = p + (n + 1) * 3

                try:
                    peak_index_range = every3(list(range(p_low, p_top)))

                    # Append peak detail lists:
                    peak_index_subsets.append(peak_index_range)
                    ecg_subsets.append(list(ecg[0][peak_index_range]))
                    peak_labels.append(lead_labels[0][p])

                except IndexError:
                    continue

        # Transform/normalise signal to be within [-1,1]:
        ecg_max = np.amax(ecg[0])
        ecg_min = np.amin(ecg[0])
        ecg_mean = np.mean(ecg[0])

        ecg_subsets = (ecg_subsets - ecg_mean) / (ecg_max - ecg_min)

        all_peak_index_subsets.append(peak_index_subsets)
        all_ecg_subsets.append(ecg_subsets)
        all_peak_labels.append(peak_labels)

    # Reshape and convert to np.array:
    flat_data = []
    for sublist in all_ecg_subsets:
        for item in sublist:
            flat_data.append(item)
    flat_data = np.asarray(flat_data)

    flat_labels = []
    for sublist in all_peak_labels:
        for item in sublist:
            flat_labels.append(item)
    flat_labels = np.asarray(flat_labels)

    return flat_data, flat_labels


# 4. Cross Validation --------------------------------------------------------

# Automated validation process:
m_range = [55, 65, 75]
layer_count_range = [400, 600, 800]  # 1600 is very slow...
epochs = 3


def run_model(m_range, layer_count_range, epochs):
    """Run model with a range of parameters/epoch."""
    # Set up validation df for later:
    validation_df = pd.DataFrame(columns={'m',
                                          'layer_neuron_count',
                                          'loss',
                                          'accuracy',
                                          'None_wrong',
                                          'P_wrong',
                                          'QRS_wrong',
                                          'T_wrong',
                                          'XtraSys_wrong'})

    for m in m_range:

        # TRAIN set
        train_leads_indices = range(len(train_leads))
        flat_train_data, flat_train_labels = preprocess(train_leads_indices, m)

        # CV set - Need to prep validation/test sets similarly to train:
        valid_leads_indices = range(len(valid_leads))
        flat_valid_data, flat_valid_labels = preprocess(valid_leads_indices, m)

        # TEST set
        test_leads_indices = range(len(test_leads))
        flat_test_data, flat_test_labels = preprocess(test_leads_indices, m)

        for n in layer_count_range:

            # Input Layer - m*2+1 input features
            # Output layer - 5 output neurons (one for each class)
            OutputLayerSize = 5
            # Hidden layer(s) neuron count - decided to use 2 layers  for now
            HiddenLayerSize = n

            # Use sequential model design in keras:
            model = keras.Sequential([
                keras.layers.Dense(HiddenLayerSize, activation="relu"),
                keras.layers.Dense(HiddenLayerSize, activation="relu"),
                keras.layers.Dense(OutputLayerSize, activation="softmax")
            ])

            # Compile
            model.compile(optimizer="SGD",
                          loss="sparse_categorical_crossentropy",
                          metrics=["accuracy"]
                          )

            # Train
            model.fit(x=flat_train_data,
                      y=flat_train_labels,
                      epochs=epochs)

            # CV
            # Making predictions:
            valid_loss, valid_acc = model.evaluate(flat_valid_data,
                                                   flat_valid_labels
                                                   )

            # Checking prediction accuracy by hand...
            prediction_prob = model.predict(flat_valid_data)

            valid_prediction = []
            for row in range(len(prediction_prob)):
                pred = np.argmax(prediction_prob[row])
                valid_prediction.append(pred)

            correct_preds = 0
            incrct_preds = pd.DataFrame(columns=['Prediction',
                                                 'Label'])
            for p in range(len(valid_prediction)):
                if valid_prediction[p] == flat_valid_labels[p]:
                    correct_preds += 1
                else:
                    incrct_preds = incrct_preds.append({'Prediction': valid_prediction[p],
                                                        'Label': flat_valid_labels[p]},
                                                       ignore_index=True)

            # Compute label specific accuracy (can vectorise later...):
            # 'None':
            wrong_0_bool = incrct_preds['Label'] == 0
            wrong_None = incrct_preds[wrong_0_bool]
            label_0_bool = flat_valid_labels == 0
            label_None = flat_valid_labels[label_0_bool]
            None_wrong = len(wrong_None) / len(label_None)

            # P wave:
            wrong_1_bool = incrct_preds['Label'] == 1
            wrong_P = incrct_preds[wrong_1_bool]
            label_1_bool = flat_valid_labels == 1
            label_P = flat_valid_labels[label_1_bool]
            P_wrong = len(wrong_P) / len(label_P)

            # QRS complex:
            wrong_2_bool = incrct_preds['Label'] == 2
            wrong_QRS = incrct_preds[wrong_2_bool]
            label_2_bool = flat_valid_labels == 2
            label_QRS = flat_valid_labels[label_2_bool]
            QRS_wrong = len(wrong_QRS) / len(label_QRS)

            # T wave:
            wrong_3_bool = incrct_preds['Label'] == 3
            wrong_T = incrct_preds[wrong_3_bool]
            label_3_bool = flat_valid_labels == 3
            label_T = flat_valid_labels[label_3_bool]
            T_wrong = len(wrong_T) / len(label_T)

            # XtraSys:
            wrong_4_bool = incrct_preds['Label'] == 4
            wrong_XtraSys = incrct_preds[wrong_4_bool]
            label_4_bool = flat_valid_labels == 4
            label_XtraSys = flat_valid_labels[label_4_bool]
            XtraSys_wrong = len(wrong_XtraSys) / len(label_XtraSys)

            # Put model performance results in a validation df:
            validation_df = validation_df.append({'m': m,
                                                  'layer_neuron_count': n,
                                                  'loss': valid_loss,
                                                  'accuracy': valid_acc,
                                                  'None_wrong': None_wrong,
                                                  'P_wrong': P_wrong,
                                                  'QRS_wrong': QRS_wrong,
                                                  'T_wrong': T_wrong,
                                                  'XtraSys_wrong': XtraSys_wrong},
                                                 ignore_index=True)
            return validation_df


if __name__ == "__main__":
    validation_df = run_model(m_range, layer_count_range, epochs)
