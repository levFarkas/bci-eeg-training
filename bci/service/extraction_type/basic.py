class BasicExtraction:

    def extract(self, x, y):
        x = self._reshape(x)
        return x, y

    @staticmethod
    def _reshape(array):
        return array.reshape(array.shape[0], 1, array.shape[1], array.shape[2])
