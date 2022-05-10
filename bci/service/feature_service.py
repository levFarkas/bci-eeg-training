import copy
from typing import Dict

import numpy as np
from mne.decoding import CSP
from sklearn.model_selection import train_test_split

from bci.model.eeg_data import EEGData, BOTH_FISTS_AND_BOTH_FEET_EXERCISES


class FeatureService:

    def feature_extraction(self, eeg_tuple: (EEGData, EEGData), csp: bool) -> tuple:
        result_x, result_y = [], []
        for eeg_data in eeg_tuple:
            y = eeg_data.epochs.events[:, -1]

            epochs_data = eeg_data.epochs.get_data()
            x = self._normalize_data(epochs_data)
            if csp:
                csp = CSP(n_components=4, log=True, reg=None)

                x = csp.fit_transform(epochs_data, y)

            result_x.append(x)
            result_y.append(y)

        return np.concatenate(tuple([*result_x])), np.concatenate(tuple([*result_y]))

    @staticmethod
    def get_test_and_train_data(featured_data: Dict[str, tuple]):
        X = [value[0] for value in featured_data.values()]
        Y = [value[1] for value in featured_data.values()]
        return train_test_split(X, Y, train_size=0.80, test_size=0.20)

    @staticmethod
    def _normalize_data(epochs_data: np.ndarray):
        new_epochs_data = copy.deepcopy(epochs_data)
        x, y, z = new_epochs_data.shape
        for i in range(z):
            for j in range(y):
                row = new_epochs_data[:, j, i]
                normalized_row = row / np.linalg.norm(row)
                # normalized_row = row / np.std(row, axis=0)
                new_epochs_data[:, j, i] = normalized_row

        return new_epochs_data
