import numpy as np
from matplotlib import pyplot as plt
from scipy import signal


class MorletCWTExtraction:

    @staticmethod
    def extract(epochs):
        sig_np = epochs.get_data()
        signal_time = sig_np.shape[2]
        channel_size = sig_np.shape[1]
        width = np.arange(1, signal_time + 1)

        all_cwt_images = []
        for j in range(sig_np.shape[0]):
            all_inner_cwt_images = []
            for i in range(channel_size):
                sig = sig_np[j, i, :]
                cwtm = abs(signal.cwt(sig, signal.morlet2, width))
                all_inner_cwt_images.append(cwtm)
                plt.pcolormesh(np.abs(cwtm), cmap='viridis', shading='gouraud')
                plt.show()
            all_cwt_images.append(np.array(all_inner_cwt_images))
        result = np.array(all_cwt_images)
        return result
