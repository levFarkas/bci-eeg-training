from typing import Dict

import numpy as np
from mne.decoding import CSP
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.backend import l2_normalize

from bci.model.eeg_data import EEGData, BOTH_FISTS_AND_BOTH_FEET_EXERCISES


class FeatureService:

    @staticmethod
    def feature_extraction(eeg_tuple: (EEGData, EEGData), csp: bool) -> tuple:
        result_x, result_y = [], []
        for eeg_data in eeg_tuple:
            y = eeg_data.epochs.events[:, -1]
            epochs_data = eeg_data.epochs.get_data()
            x = _normalize_data(epochs_data)
            if csp:
                csp = CSP(n_components=4, log=True, reg=None)

                x = csp.fit_transform(epochs_data, y)

            result_x.append(x)
            result_y.append(y)

        x = np.concatenate(tuple([*result_x]))
        x = reshape(x)
        y = np.concatenate(tuple([*result_y]))

        return x, y

    @staticmethod
    def get_test_and_train_data(featured_data: Dict[str, tuple]):
        X = [value[0] for value in featured_data.values()]
        Y = [value[1] for value in featured_data.values()]
        return train_test_split(X, Y, train_size=0.80, test_size=0.20)


def reshape(array):
    return array.reshape(array.shape[0], 1, array.shape[1], array.shape[2])


def _normalize_data(epochs_data: np.ndarray):

    new_epochs_data = l2_normalize(epochs_data, 0)
    new_epochs_data = l2_normalize(new_epochs_data, 1)

    return new_epochs_data
    # x, y, z = new_epochs_data.shape
    # for i in range(x):
    #     for j in range(y):
    #         row = new_epochs_data[i, j, :]
    #         # normalized_row = row / np.linalg.norm(row)
    #         # normalized_row = (row - np.average(row)) / np.std(row, axis=0)
    #
    #         # normalized_row = row / np.std(row, axis=0)
    #         new_epochs_data[i, j, :] = normalized_row

