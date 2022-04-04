from keras.models import Input


class NeuralNetwork:

    def train(self, train_data, test_data):
        batch_size, timesteps, input_dim = None, 20, 1

        i = Input(batch_shape=(batch_size, timesteps, input_dim))
