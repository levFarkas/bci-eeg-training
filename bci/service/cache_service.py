import pickle
from pathlib import Path
from typing import Dict, Tuple

from bci.model.eeg_data import EEGData


class CacheService:

    def __init__(self):
        self._cache_path = "../resources/eeg_data.pickles"
        self._plot_data = False

    def cache_data(self, data: object):
        with open(self._cache_path, 'wb') as f:
            pickle.dump(data, f)

    def is_already_cached(self) -> bool:
        path = Path(self._cache_path)
        return path.is_file()

    def get_cached_data(self) -> Dict[str, Tuple[EEGData, EEGData]]:
        with open(self._cache_path, 'rb') as f:
            data = pickle.load(f)
            if self._plot_data:
                self._plot(data)
            return data

    @staticmethod
    def _plot(data: Dict[str, Tuple[EEGData, EEGData]]):
        for key, value in data.items():
            print(f"Patient: {key}")
            for eeg_data in value:
                eeg_data.raw.plot(n_channels=64)
