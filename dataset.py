import os
from dataclasses import dataclass
from typing import List, Optional
from utils import load_yaml
from enum import Enum


class DatasetPart(Enum):
    TRAIN = 'train'
    VAL = 'val'
    TEST = 'test'
    DATA = 'data'


@dataclass
class DatasetConfig:
    train: Optional[str]
    val: Optional[str]
    test: Optional[str]
    data: Optional[str]
    nc: int
    names: List[str]

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


class Dataset:

    def _safe_load_config(self):
        config_file_path = os.path.join(self.dataset_folder_path)

        if not os.path.isfile(config_file_path):
            raise FileNotFoundError(f'Config file not found at {config_file_path}')

        config = load_yaml(config_file_path)
        self.config = DatasetConfig(**config)

    def __init__(self, dataset_folder_path):
        self.dataset_folder_path = dataset_folder_path
        self.config = None
        self._safe_load_config()

    def iterate(self, part: DatasetPart):
        if not self.config[part]:
            raise ValueError(f'No path provided for {part}')
