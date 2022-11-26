import numpy as np
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import ShuffleSplit, cross_val_score
from sklearn.pipeline import Pipeline


class CSPExtraction:

    def extract(self, epochs):
        epochs_data = epochs
        epochs_train = epochs.copy().crop(tmin=0.2, tmax=2)
        labels = epochs_data.events[:, -1] - 2
        epochs_data_train = epochs_train.get_data()
        cv = ShuffleSplit(10, test_size=0.2, random_state=42)

        # Assemble a classifier
        lda = LinearDiscriminantAnalysis()
        csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)

        # Use scikit-learn Pipeline with cross_val_score function
        clf = Pipeline([('CSP', csp), ('LDA', lda)])
        scores = cross_val_score(clf, epochs_data_train, labels, cv=cv, n_jobs=None)

        # Printing the results
        class_balance = np.mean(labels == labels[0])
        class_balance = max(class_balance, 1. - class_balance)
        print("Classification accuracy: %f / Chance level: %f" % (np.mean(scores),
                                                                  class_balance))

        # plot CSP patterns estimated on full data for visualization
        csp.fit_transform(epochs_data.get_data(), labels)

        csp.plot_patterns(epochs.info, ch_type='eeg', units='Patterns (AU)', size=1.5)
