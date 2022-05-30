from dataclasses import dataclass

from mne import Epochs
from mne.io.edf.edf import RawEDF

ANNOTATIONS_LEFT_AND_RIGHT_HANDS = {
    "T0": 1,
    "T1": 2,
    "T2": 3
}

ANNOTATIONS_BOTH_FISTS_AND_FEET = {
    "T0": 1,
    "T1": 4,
    "T2": 5
}

LEFT_AND_RIGHT_HAND_EXERCISES = [3, 4, 7, 8, 11, 12]
BOTH_FISTS_AND_BOTH_FEET_EXERCISES = [5, 6, 9, 10, 13, 14]


@dataclass
class EEGData:
    exercise_number: int
    record_name: str
    raw: RawEDF
    events: tuple = None
    epochs: Epochs = None
