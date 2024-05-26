from typing import Dict, Tuple

import numpy as np

from bci.loaders.load_engine import LoadEngine
from bci.model.eeg_data import EEGData
from bci.neural_network import NeuralNetwork
from bci.service.cache_service import CacheService
from bci.service.feature_service import FeatureService
from bci.service.plotter import Plotter
from bci.service.trainer import Trainer


class Handler:
    EXTRACTION_TYPE = "all"
    SHOW_ONLINE = True
    CUT_STEP = 0.25

    def __init__(self):
        self._cache_service = CacheService()
        self._load_engine = LoadEngine(cache_service=self._cache_service)
        self._feature_service = FeatureService()
        self._trainer = Trainer()
        self._plotter = Plotter()
        self._accuracies = {
            "csp": [],
            "cnn": [],
            "morlet": []
        }

    def process(self, cut=0.0) -> None:
        data = self._cache_service.get_cached_data() \
            if self._cache_service.is_already_cached() \
            else self._load_engine.load_all_data("../resources/files/eegmmidb/1.0.0")

        if self.SHOW_ONLINE:
            if cut > 1:
                self._plotter.plot_accuracies(self._accuracies)
                return
            self._cut(data, self.CUT_STEP, cut)

            if self.EXTRACTION_TYPE == "csp" or self.EXTRACTION_TYPE == "all":
                featured_data = self._convert_to_featured_data(data, type="csp")
                csp_accuracies = [self._trainer.csp_training(v[0], v[1]) for _, v in featured_data.items()]
                csp_accuracy = np.mean(csp_accuracies)
                self._accuracies["csp"].append(csp_accuracy)
            if self.EXTRACTION_TYPE == "morlet" or self.EXTRACTION_TYPE == "all" or self.EXTRACTION_TYPE == "basic":
                featured_data = self._convert_to_featured_data(data, type="basic")
                accuracy = self._run_neural_network(featured_data, type="basic")
                self._accuracies["cnn"].append(accuracy)
                featured_data = self._convert_to_featured_data(data, type="morlet")
                accuracy = self._run_neural_network(featured_data, type="morlet")
                self._accuracies["morlet"].append(accuracy)
            self.process(cut=cut + self.CUT_STEP)
        else:
            featured_data = self._convert_to_featured_data(data, type=self.EXTRACTION_TYPE)

            if self.EXTRACTION_TYPE == "csp":
                for _, v in featured_data.items():
                    self._trainer.csp_training(v[0], v[1])

            self._run_neural_network(featured_data)

    def _convert_to_featured_data(self, data: Dict[str, Tuple[EEGData, EEGData]], **kwargs) -> Dict[str, tuple]:
        return {key: self._feature_service.feature_extraction(data[key], type=kwargs.get("type")) for key in data}

    @staticmethod
    def flatten(l):
        return [item for sublist in l for item in sublist]

    def _run_neural_network(self, featured_data: Dict[str, tuple], type: str = "basic") -> float:
        train_features, test_features, train_labels, test_labels = self._feature_service.get_test_and_train_data(
            featured_data)

        neural_network = NeuralNetwork()

        train_labels = np.concatenate(train_labels)
        train_features = np.concatenate(train_features)
        test_labels = np.concatenate(test_labels)
        test_features = np.concatenate(test_features)

        if type == "basic":
            return neural_network.train(
                train_labels=train_labels,
                train_features=train_features,
                test_labels=test_labels,
                test_features=test_features
            )
        if type == "morlet":
            return neural_network.train_morlet(
                train_labels=train_labels,
                train_features=train_features,
                test_labels=test_labels,
                test_features=test_features
            )
        return 0.0

    @staticmethod
    def _cut(data: Dict[str, Tuple[EEGData, EEGData]], cut_range, offset):
        times = {k: v[0].epochs.times for k, v in data.items()}

        new_data = {k: v[0].epochs.crop(tmin=times[k].min() * offset, tmax=times[k].max() * cut_range) for k, v in
                    data.items()}
        return new_data


if __name__ == '__main__':
    handler = Handler()
    handler.process()
