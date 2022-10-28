import random
from typing import Dict, Tuple

import numpy as np
from mne.decoding import CSP
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.backend import l2_normalize

from bci.model.eeg_data import EEGData


class FeatureService:

    @staticmethod
    def _shuffle_two_arrays_in_same_order(arr1, arr2):
        c = list(zip(list(arr1), list(arr2)))
        random.shuffle(c)
        arr1_res, arr2_res = zip(*c)
        return np.array(arr1_res), np.array(arr2_res)

    def feature_extraction(self, eeg_tuple: (EEGData, EEGData), csp: bool) -> tuple:
        result_x, result_y = [], []

        for eeg_data in eeg_tuple:
            y = eeg_data.epochs.events[:, -1]
            epochs_data = eeg_data.epochs.get_data()
            x = _normalize_data(epochs_data)
            result_x.append(x)
            result_y.append(y)

        x = np.concatenate(tuple([*result_x]))
        y = np.concatenate(tuple([*result_y]))

        if csp:
            # y_unique = np.unique(y)
            # TODO - do handshake algorithm
            y_pairs = ((1, 2), (1, 3), (2, 3))
            x_results = []
            for pair in y_pairs:
                x_csp, y_csp = self._get_prepared_csp_data_based_on_pairs(pair, x, y)
                x_csp, y_csp = self._shuffle_two_arrays_in_same_order(x_csp, y_csp)

                csp = CSP(n_components=4, log=True, reg=None)
                if len(x_csp) % 2 == 1:
                    fit_set, transform_set = np.split(x_csp[0:-1, :, :], 2, 0)
                    fit_label, _ = np.split(y_csp[0:-1], 2, 0)
                else:
                    fit_set, transform_set = np.split(x_csp, 2, 0)
                    fit_label, _ = np.split(y_csp, 2, 0)

                x_csp_result = csp.fit_transform(fit_set, fit_label) + csp.transform(transform_set)
                x_results.append(x_csp_result)
            x = np.concatenate(tuple([*x_results]))
        # x = reshape(x)
        return x, y

    @staticmethod
    def get_test_and_train_data(featured_data: Dict[str, tuple], train_size=0.80):
        x = [value[0] for value in featured_data.values()]
        y = [value[1] for value in featured_data.values()]
        return train_test_split(x, y, train_size=train_size, test_size=1 - train_size)

    @staticmethod
    def _get_prepared_csp_data_based_on_pairs(pair: Tuple, x, y) -> Tuple:
        y1, y2 = pair[0], pair[1]
        y1_indices = [i for i, x in enumerate(list(y)) if x == y1]
        y2_indices = [i for i, x in enumerate(list(y)) if x == y2]

        x1 = np.array([x for i, x in enumerate(list(x)) if i in y1_indices])
        x2 = np.array([x for i, x in enumerate(list(x)) if i in y2_indices])

        y1_np_array = np.array([y1 for i in list(x1)])
        y2_np_array = np.array([y2 for i in list(x2)])

        return np.concatenate((x1, x2)), np.concatenate((y1_np_array, y2_np_array))


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
