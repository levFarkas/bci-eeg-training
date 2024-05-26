import numpy as np
from scipy import signal
from sklearn import preprocessing

from bci.service.cache_service import CacheService


class MorletCWTExtraction:

    def __init__(self):
        self._cacheService = CacheService()

    def extract(self, epochs):
        sig_np = epochs.get_data()
        channel_size = sig_np.shape[1]
        width = np.arange(1, 31)

        all_cwt_images = []
        for j in range(sig_np.shape[0]):
            all_inner_cwt_images = []
            for i in range(channel_size):
                sig = sig_np[j, i, :]
                cwtm = abs(signal.cwt(sig, signal.morlet2, width))
                cwtm = preprocessing.normalize(cwtm, norm='l2')
                all_inner_cwt_images.append(cwtm)
            all_cwt_images.append(np.array(all_inner_cwt_images))
        result = np.array(all_cwt_images)
        # self._cacheService.cache_data("../resources/morlet_cache.pickles")

        return result
