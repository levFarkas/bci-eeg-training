import random
from typing import Dict, Tuple

import numpy as np
from keras.backend import l2_normalize
from sklearn.model_selection import train_test_split

from bci.model.eeg_data import EEGData
from bci.service.extraction_type.basic import BasicExtraction
from bci.service.extraction_type.csp import CSPExtraction
from bci.service.extraction_type.morlet_cwt import MorletCWTExtraction


class FeatureService:

    @staticmethod
    def _shuffle_two_arrays_in_same_order(arr1, arr2):
        c = list(zip(list(arr1), list(arr2)))
        random.shuffle(c)
        arr1_res, arr2_res = zip(*c)
        return np.array(arr1_res), np.array(arr2_res)

    @staticmethod
    def feature_extraction(eeg_tuple: (EEGData, EEGData), **kwargs) -> tuple:
        result_x, result_y = [], []

        for eeg_data in eeg_tuple:
            y = eeg_data.epochs.events[:, -1]
            epochs_data = eeg_data.epochs.get_data()
            x = _normalize_data(epochs_data)
            result_x.append(x)
            result_y.append(y)

        x = np.concatenate(tuple([*result_x]))
        y = np.concatenate(tuple([*result_y]))

        extraction_type = kwargs.get("type")

        if extraction_type == "basic":
            extraction = BasicExtraction()
            return extraction.extract(x, y)
        elif extraction_type == "csp":
            extraction = CSPExtraction()
            return extraction.extract(eeg_tuple[0].epochs)
        elif extraction_type == "morlet":
            extraction = MorletCWTExtraction()
            return extraction.extract(eeg_tuple[0].epochs)

        raise RuntimeError("Not supported extractionType")

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

        y1_np_array = np.array([y1 for _ in list(x1)])
        y2_np_array = np.array([y2 for _ in list(x2)])

        return np.concatenate((x1, x2)), np.concatenate((y1_np_array, y2_np_array))


def _normalize_data(epochs_data: np.ndarray):
    new_epochs_data = l2_normalize(epochs_data, 0)
    new_epochs_data = l2_normalize(new_epochs_data, 1)

    return new_epochs_data
