import mne
import tensorflow as tf
from sklearn.model_selection import KFold
import numpy as np
import random


class Network:
    def __init__(self, x_shape, n, params):
        self.model = tf.keras.Sequential()
        self.time_steps = x_shape[1]
        self.data_points = x_shape[0]
        self.num_epochs = n
        self.create_model(params)
        self.callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, mode='auto')


    def create_model(self, params):

        self.model.add(tf.keras.layers.Conv2D(params['layer1'], kernel_size=(15, 1),
                       input_shape=(1, self.data_points, self.time_steps),
                       padding='same'))
        self.model.add(tf.keras.layers.Dropout(0.25))
        self.model.add(tf.keras.layers.Activation('relu'))

        self.model.add(tf.keras.layers.Conv2D(params['layer2'], kernel_size=(1, self.data_points),
                                              input_shape=(1, self.data_points, self.time_steps),
                                              padding='valid'))
        self.model.add(tf.keras.layers.Dropout(0.25))
        self.model.add(tf.keras.layers.Activation('relu'))

        self.model.add(tf.keras.layers.AveragePooling2D(pool_size=(15, 1), padding='same'))
        self.model.add(tf.keras.layers.Dropout(0.25))
        self.model.add(tf.keras.layers.Flatten())
        self.model.add(tf.keras.layers.Dense(params['neurons']))
        self.model.add(tf.keras.layers.Activation('relu'))
        self.model.add(tf.keras.layers.Activation('softmax'))
        self.model.add(tf.keras.layers.Dropout(0.25))
        self.model.add(tf.keras.layers.Dense(4))
        self.model.add(tf.keras.layers.Activation('softmax'))

    def train(self, x, y, val_x, val_y, summary):
        self.model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
        self.model.fit(x, y, batch_size=16, epochs=self.num_epochs, verbose=1, validation_data=(val_x, val_y),
                       callbacks=[self.callback])
        if summary:
            self.model.summary()

    def eval(self, x, y):
        return self.model.evaluate(x, y, verbose=0)

    def cross_val(self, num_folds, subjects, records):
        (m, all_x, n, all_y) = get_data(subjects, records, 0, (88, 89, 92, 100))
        kfold = KFold(n_splits=num_folds, shuffle=True)
        fold_nr = 0

        accuracy = []
        loss = []
        models = []
        for train, test in kfold.split(all_x, all_y):
            print(f'Current fold: {fold_nr}')
            self.model = tf.keras.Sequential()
            self.create_model({'layer1': 40, 'layer2': 40, 'neurons': 80})
            self.train(all_x[train], all_y[train], all_x[test], all_y[test], 0)
            fold_nr += 1
            scores = self.model.evaluate(all_x[test], all_y[test], verbose=0)
            accuracy.append(scores[1])
            loss.append(scores[0])
            models.append(self.model)

        i = maxindex(accuracy)
        models[i].save(f'../output/model_acc_{accuracy[i]}')


def getfile(subject, record):
    a = '{:0>3}'.format(subject)
    b = '{:0>2}'.format(record)
    filename = f'../resources/files/eegmmidb/1.0.0/S{a}/S{a}R{b}.edf'

    raw = mne.io.read_raw_edf(filename, verbose=0)
    (events, event_id) = mne.events_from_annotations(raw, verbose=0)

    event_dict = {
        'REST': 1,
        'LEFT': 2,
        'RIGHT': 3
    }

    epochs = mne.Epochs(raw, events, event_id=event_dict, tmin=-0.2, tmax=0.5, on_missing='ignore', verbose=0)
    epochs.equalize_event_counts()
    data = epochs.get_data()
    y = epochs.events[:, 2]
    return data, y


def get_data(subjects, records, split_where, drop):

    all_data = []
    all_y = []
    num_subjects = 0
    for j in subjects:
        subject_data = []
        subject_y = []
        if j in drop:
            continue
        for i in records:
            subject_data.append(getfile(j, i)[0])
            subject_y.append(getfile(j, i)[1])
        all_data.append(np.concatenate(subject_data))
        all_y.append(np.concatenate(subject_y))
        print(f'Reading subject {j}')
        num_subjects += 1

    (x, y) = (np.concatenate(all_data), np.concatenate(all_y))
    x = tf.math.l2_normalize(x, 0).numpy()
    x = tf.math.l2_normalize(x, 1).numpy()
    (x_tr, x_te) = split(x, split_where)
    (y_tr, y_te) = split(y, split_where)
    x_tr = reshape(x_tr)
    x_te = reshape(x_te)
    y_tr = tf.keras.utils.to_categorical(y_tr, 4)
    y_te = tf.keras.utils.to_categorical(y_te, 4)

    print(f'{num_subjects} subjects have been read')
    return x_tr, x_te, y_tr, y_te


def split(data, where):
    where = int(data.shape[0]*where)
    return data[:where], data[where:]


def reshape(array):
    return array.reshape(array.shape[0], 1, array.shape[1], array.shape[2])


def maxindex(x):
    index = 0
    max = 0
    i = 0
    for j in x:
        if j > max:
            max = j
            index = i
        i += 1
    return index


def GridSearch(params, x_t, x_val, y_t, y_val, x_test, y_test):
    scores = []
    models = []
    opt_p = []
    n = 0

    for i in params['layer1']:
        for k in params['layer2']:
            for m in params['neurons']:
                print(f'Progress: {n}')
                netw = Network([64, 113], 25, {'layer1': i, 'layer2': k, 'neurons': m})
                netw.train(x_t, y_t, x_val, y_val, 0)
                scores.append(netw.eval(x_test, y_test))
                models.append(netw.model)
                n += 1
                opt_p.append({'layer1': i, 'layer2': k, 'neurons': m})

    return scores, models

def RandomSearch(params, x_t, x_val, y_t, y_val, x_test, y_test, n):
    scores = []
    models = []
    opt_p = []
    p = 0

    for i in range(n):
        print(f'Progress: {p}')
        a = random.choice(params['layer1'])
        b = random.choice(params['layer2'])
        c = random.choice(params['neurons'])
        par = {'layer1': a, 'layer2': b, 'neurons': c}

        netw = Network([64, 113], 25, par)
        netw.train(x_t, y_t, x_val, y_val, 0)
        scores.append(netw.eval(x_test, y_test))
        models.append(netw.model)
        opt_p.append(par)
        p+= 1

    return scores, models, opt_p

if __name__ == '__main__':

    (x_train, x_val, y_train, y_val) = get_data(range(1, 5), (3, 4, 7, 8, 11, 12), 0.8, (88, 89, 92, 100))
    (tmp1, x_test, tmp2, y_test) = get_data((4, 5), (3, 4, 7, 8, 11, 12), 0, (89, 88, 92, 100))
    params = {
        'layer1': [20, 30, 40, 50],
        'layer2': [20, 30, 40, 50],
        'neurons': [60, 70, 75, 80, 90, 100, 120]
    }

    # s, m, o = GridSearch(params, x_train, x_val, y_train, y_val, x_test, y_test)
    s, m, o = RandomSearch(params, x_train, x_val, y_train, y_val, x_test, y_test, 2)

    print(s)
    print(len(m))
    print(o)