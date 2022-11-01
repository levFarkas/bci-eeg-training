import itertools

import matplotlib.pyplot as plt
import numpy as np
from keras import Sequential
from keras.callbacks import EarlyStopping
from keras.layers import Conv2D, Flatten, Dense, Activation, MaxPooling2D, \
    BatchNormalization, Dropout, AveragePooling2D
from keras.optimizer_v2.adam import Adam
from sklearn.metrics import confusion_matrix


class NeuralNetwork:

    def train(self, train_features, test_features, train_labels, test_labels):
        params = {
            'layer1': 16,
            'layer2': 32,
            'neurons': 30
        }

        x_shape = [64, 113]
        data_points = x_shape[0]
        time_steps = x_shape[1]

        model = Sequential([
            Conv2D(params['layer1'], kernel_size=(15, 1),
                   input_shape=(1, data_points, time_steps),
                   padding='same'),
            Dropout(0.25),
            Activation("relu"),
            Conv2D(params['layer2'], kernel_size=(1, data_points),
                   input_shape=(1, data_points, time_steps),
                   padding='valid'),
            Dropout(0.25),
            Activation("relu"),
            AveragePooling2D(pool_size=(15, 1), padding='same'),
            Dropout(0.25),
            Flatten(),
            Dense(params['neurons']),
            Activation('relu'),
            Activation('softmax'),
            Dropout(0.25),
            Dense(4),
            Activation('softmax')
        ])

        model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        model.summary()

        es_callback = EarlyStopping(monitor='val_loss', patience=5, mode="auto")

        model.fit(x=train_features, y=train_labels, validation_split=0.2, batch_size=16,
                  epochs=75, verbose=1, callbacks=[es_callback])

        cm_plot_labels = ["Rest", "Left Hand", "Right Hand"]
        predictions = model.predict(x=test_features, batch_size=5, verbose=0)
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
