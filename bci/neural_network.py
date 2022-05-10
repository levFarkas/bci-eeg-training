import itertools

import matplotlib.pyplot as plt
import numpy as np
from keras import Sequential
from keras.layers import Conv2D, AveragePooling2D, Flatten, Dense
from sklearn.metrics import confusion_matrix


class NeuralNetwork:

    def train(self, train_features, test_features, train_labels, test_labels):
        model = Sequential([
            Conv2D(filters=16, kernel_size=[1, 30], input_shape=(64, 113, 1), activation='relu'),
            Conv2D(filters=32, kernel_size=[64, 1], activation='relu'),
            AveragePooling2D(pool_size=(1, 2)),
            Flatten(),
            Dense(units=2, activation='softmax')
        ])

        # labels = {}
        # for label in train_labels:
        #     if label in labels:
        #         labels[label] += 1
        #     else:
        #         labels[label] = 1

        model.summary()
        model.compile(optimizer="adam", loss='sparse_categorical_crossentropy', metrics=['accuracy'])
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
