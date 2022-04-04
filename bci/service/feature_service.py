import itertools

import numpy as np
import matplotlib.pyplot as plt

from typing import Dict

from keras import Sequential
from keras.layers import Dense, Conv2D, AveragePooling2D, Flatten
from keras.optimizer_v1 import Adam
from mne.decoding import CSP
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from bci.model.eeg_data import EEGData, BOTH_FISTS_AND_BOTH_FEET_EXERCISES


class FeatureService:

    def feature_extraction(self, eeg_tuple: (EEGData, EEGData)) -> tuple:
        result_x, result_y = [], []
        for eeg_data in eeg_tuple:
            y = eeg_data.epochs.events[:, -1] - 2

            if eeg_data.exercise_number in BOTH_FISTS_AND_BOTH_FEET_EXERCISES:
                y += 2

            epochs_data = eeg_data.epochs.get_data()
            csp = CSP(n_components=4, log=True, reg=None)

            x = csp.fit_transform(epochs_data, y)

            result_x.append(x)
            result_y.append(y)

        return np.concatenate(tuple([*result_x])), np.concatenate(tuple([*result_y]))

    def get_test_and_train_data(self, featured_data: Dict[str, tuple]):
        X = [value[0] for value in featured_data.values()]
        Y = [value[1] for value in featured_data.values()]
        train_features, test_features, train_labels, test_labels = \
            train_test_split(X, Y, train_size=0.80, test_size=0.20)

        self._train(train_features, test_features, train_labels, test_labels)

    def _train(self, train_features, test_features, train_labels, test_labels):

        model = Sequential([
            Conv2D(filters=16, kernel_size=[30, 1], input_shape=(384, 4, 1), activation='relu'),
            Conv2D(filters=32, kernel_size=[1, 1526], input_shape=(384, 4, 1), activation='relu'),
            AveragePooling2D(),
            Flatten(),
            Dense(units=2, activation='softmax')
        ])

        model.summary()

        model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        model.fit(x=train_features, y=train_labels, validation_data=(test_features, test_labels), batch_size=10,
                  epochs=75, shuffle=True, verbose=2)

        cm_plot_labels = ['Left Hand', 'Right Hand']
        predictions = model.predict(x=test_features, batch_size=10, verbose=0)
        rounded_predictions = np.argmax(predictions, axis=-1)
        cm = confusion_matrix(y_true=test_labels, y_pred=rounded_predictions)
        self.plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title="Confusion Matrix")
        plt.show()

    @staticmethod
    def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion Matrix', cmap=plt.cm.Blues):
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print("Confusion matrix, without normalization")

        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, cm[i, j],
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()
