import itertools
from typing import List, Dict, Tuple

import mne
from mne import Epochs, events_from_annotations, pick_types, concatenate_raws
from mne.datasets.eegbci import eegbci
from mne.io import read_raw_edf
from mne.preprocessing import ICA

from bci.model.eeg_data import EEGData, ANNOTATIONS_LEFT_AND_RIGHT_HANDS, BOTH_FISTS_AND_BOTH_FEET_EXERCISES, LEFT_AND_RIGHT_HAND_EXERCISES, \
    ANNOTATIONS_BOTH_FISTS_AND_FEET
from bci.service.cache_service import CacheService

EXCLUDED_PATIENTS = [88, 89, 92, 100]


class LoadEngine:

    def __init__(self, cache_service: CacheService):
        self._cache_service = cache_service

    def load_all_data(self, folder_path: str) -> Dict[str, Tuple[EEGData, EEGData]]:
        raw_data = self._load_raw_data(folder_path)

        # TODO remove test parameter

        test = True

        aggregated_data = self._aggregate_data(raw_data, test)
        self._preprocess(aggregated_data)
        self._cache_service.cache_data(aggregated_data)
        return aggregated_data

    def _load_raw_data(self, folder_path: str) -> List[EEGData]:
        records = [record.strip() for record in self._get_records(folder_path)]
        all_data = []
        for record in records:
            # get RXX value of the record
            sample = int(record[-6:-4])
            # get SXX value of the record
            record_name = record[:4]
            if int(record_name[1:]) > 5:
                continue

            if int(record_name[1:]) in EXCLUDED_PATIENTS:
                continue

            file_path = folder_path + "/" + record
            raw = read_raw_edf(file_path, preload=True)
            all_data.append(
                EEGData(
                    exercise_number=sample,
                    record_name=record_name,
                    raw=raw
                )
            )
        return all_data

    @staticmethod
    def _aggregate_data(data: List[EEGData], test=False) -> Dict[str, Tuple[EEGData, EEGData]]:

        def multiple_eeg_data_to_simple_one(eeg_list: List[EEGData]) -> (EEGData, EEGData):

            both_hands_and_feet_raws = [eeg.raw for eeg in eeg_list if
                                        eeg.exercise_number in BOTH_FISTS_AND_BOTH_FEET_EXERCISES]
            left_and_right_hand_raws = [eeg.raw for eeg in eeg_list if
                                        eeg.exercise_number in LEFT_AND_RIGHT_HAND_EXERCISES]

            if test:
                return (
                    EEGData(
                        exercise_number=LEFT_AND_RIGHT_HAND_EXERCISES[0],
                        record_name=eeg_list[0].record_name,
                        raw=concatenate_raws(left_and_right_hand_raws)
                    ),
                )

            return (
                EEGData(
                    exercise_number=BOTH_FISTS_AND_BOTH_FEET_EXERCISES[0],
                    record_name=eeg_list[0].record_name,
                    raw=concatenate_raws(both_hands_and_feet_raws)
                ),
                EEGData(
                    exercise_number=LEFT_AND_RIGHT_HAND_EXERCISES[0],
                    record_name=eeg_list[0].record_name,
                    raw=concatenate_raws(left_and_right_hand_raws)
                )
            )

        iterator = itertools.groupby(data, lambda x: x.record_name[0:4])
        aggregated_data = {}
        for key, group in iterator:
            aggregated_data[key] = multiple_eeg_data_to_simple_one(list(group))

        return aggregated_data

    @staticmethod
    def _get_records(folder_path: str) -> List[str]:
        """

        To get all records that represent the edf files and event

        :param folder_path: path to RECORDS file
        :return: all records included in RECORDS file
        """
        lines = []
        with open(folder_path + "/RECORDS", "r") as f:
            for line in f:
                lines.append(line)
        return lines

    @staticmethod
    def _preprocess(aggregated_data: Dict[str, Tuple[EEGData, EEGData]]):
        """

        - band pass filter raw data (7 Hz to 30 Hz)
        - save epochs and events to the model
        - fastica in order to improve classification rate.

        :param aggregated_data: Dict[str: EEGData]
        :return: None
        """

        for eeg_tuple in aggregated_data.values():
            for eeg_data in eeg_tuple:
                raw_data = eeg_data.raw
                raw_data.rename_channels(lambda x: x.strip('.'))
                eegbci.standardize(raw_data)
                montage = mne.channels.make_standard_montage('standard_1005')
                no_position = [ch for ch in raw_data.copy().pick_types(eeg=True).ch_names if ch not in montage.ch_names]

                raw_data.set_montage(montage, on_missing='warn')
                raw_data = raw_data.pick(None, exclude=no_position)

                raw_data.filter(1., 50., fir_design='firwin')
                # raw_data.filter(7., 30., method="iir")
                # raw_data = preprocessing.normalize(raw_data, norm='l2')
                picks = pick_types(raw_data.info, meg=False, eeg=True, stim=False, eog=False, exclude="bads")

                if eeg_data.exercise_number in BOTH_FISTS_AND_BOTH_FEET_EXERCISES:
                    events = events_from_annotations(raw_data, ANNOTATIONS_BOTH_FISTS_AND_FEET)
                else:
                    events = events_from_annotations(raw_data, ANNOTATIONS_LEFT_AND_RIGHT_HANDS)
                epochs = Epochs(raw_data, events[0], tmin=0, tmax=2, picks=picks, preload=True, baseline=None)
                epochs.equalize_event_counts()
                eeg_data.epochs = epochs
                eeg_data.events = events
                ica = ICA(n_components=4, method="fastica")
                ica.fit(raw_data)

