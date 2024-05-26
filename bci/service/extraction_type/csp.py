class CSPExtraction:

    @staticmethod
    def extract(epochs):
        labels = epochs.events[:, -1] - 2
        return epochs, labels
