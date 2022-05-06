import matplotlib.pyplot as plt


from typing import Dict, Tuple

from bci.loaders.load_engine import LoadEngine
from bci.model.eeg_data import EEGData
from bci.neural_network import NeuralNetwork
from bci.service.cache_service import CacheService
from bci.service.feature_service import FeatureService


class Handler:

    def __init__(self):
        self._cache_service = CacheService()
        self._load_engine = LoadEngine(cache_service=self._cache_service)
        self._feature_service = FeatureService()

    def process(self) -> None:
        data = self._cache_service.get_cached_data() \
            if self._cache_service.is_already_cached() \
            else self._load_engine.load_all_data("../resources/files/eegmmidb/1.0.0")

        featured_data = self._convert_to_featured_data(data, csp=False)
        train_features, test_features, train_labels, test_labels = self._feature_service.get_test_and_train_data(featured_data)
        neural_network = NeuralNetwork()
        neural_network.train(
            train_labels=train_labels,
            train_features=train_features,
            test_labels=test_labels,
            test_features=test_features
        )

    def _convert_to_featured_data(self, data: Dict[str, Tuple[EEGData, EEGData]], csp: bool) -> Dict[str, tuple]:
        return {key: self._feature_service.feature_extraction(data[key], csp) for key in data}



if __name__ == '__main__':
    handler = Handler()
    handler.process()
