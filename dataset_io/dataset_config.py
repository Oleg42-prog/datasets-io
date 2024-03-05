from enum import Enum
from typing import List, Optional
from dataclasses import dataclass


class DatasetPart(Enum):
    TRAIN = 'train'
    VAL = 'val'
    TEST = 'test'
    DATA = 'data'


@dataclass
class DatasetConfig:
    nc: int
    names: List[str]
    train: Optional[str] = ''
    val: Optional[str] = ''
    test: Optional[str] = ''
    data: Optional[str] = ''

    def _at_least_one_path(self):
        return any([self.train, self.val, self.test, self.data])

    def _class_numbers_consistent(self):
        return self.nc == len(self.names)

    def __post_init__(self):
        if not self._at_least_one_path():
            raise ValueError('At least one path must be provided in the config file')

        if not self.names:
            raise ValueError('Class names must be provided in the config file')

        if not self._class_numbers_consistent():
            raise ValueError('Number of class names must match the number of classes')
