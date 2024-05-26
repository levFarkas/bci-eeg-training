from typing import Dict, List

import numpy as np
from matplotlib import pyplot as plt


class Plotter:
    @staticmethod
    def plot_accuracies(accuracies: Dict[str, List]):
        for k, v in accuracies.items():
            t = np.linspace(0, 1, num=len(v))
            plt.plot(t, v, label=k)
        plt.legend()
        plt.ylabel("Accuracy")
        plt.xlabel("Percentage of used dataset")
        plt.show()
