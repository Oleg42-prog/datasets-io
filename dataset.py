import os
from dataclasses import dataclass, asdict
from typing import List, Optional
from utils import load_yaml, load_text, read_lines, change_file_name_extension
from enum import Enum
from PIL import Image


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


class Dataset:

    def _safe_load_config(self, config_file_name):
        config_file_path = os.path.join(self.dataset_folder_path, config_file_name)

        if not os.path.isfile(config_file_path):
            raise FileNotFoundError(f'Config file not found at {config_file_path}')

        config = load_yaml(config_file_path)
        self.config = DatasetConfig(**config)
        self.config_dict = asdict(self.config)

    def __init__(self, dataset_folder_path, config_file_name='data.yaml'):
        self.dataset_folder_path = dataset_folder_path
        self.config = None
        self._safe_load_config(config_file_name)

    def _get_part_path(self, part: DatasetPart):
        part_path = self.config_dict[part.value]
        full_part_path = os.path.join(self.dataset_folder_path, part_path)

        if not part_path:
            raise ValueError(f'No path provided for {part_path}')

        if not os.path.isdir(full_part_path):
            raise FileNotFoundError(f'Part folder not found at {part_path}')

        return part_path

    def iterate_file_paths(self, part: DatasetPart, skip_missing_labels=True):
        part_path = self._get_part_path(part)

        image_folder_path = os.path.join(self.dataset_folder_path, part_path)
        if not os.path.isdir(image_folder_path):
            raise FileNotFoundError(f'Image folder not found at {image_folder_path}')

        label_folder_path = image_folder_path.replace('images', 'labels')
        if not os.path.isdir(label_folder_path):
            raise FileNotFoundError(f'Label folder not found at {label_folder_path}')

        image_file_names = sorted(os.listdir(image_folder_path))
        for image_file_name in image_file_names:
            label_file_name = change_file_name_extension(image_file_name, '.txt')

            image_file_path = os.path.join(image_folder_path, image_file_name)
            label_file_path = os.path.join(label_folder_path, label_file_name)

            if skip_missing_labels and not os.path.isfile(label_file_path):
                continue

            yield image_file_path, label_file_path

    def iterate(self, part: DatasetPart):
        raise NotImplementedError('This method is not implemented yet')

    def lazy_iterate(self, part: DatasetPart):
        for image_file_path, label_file_path in self.iterate_file_paths(part):
            image = Image.open(image_file_path)
            label = LabeledBBox.from_yolov5_file(label_file_path)
            yield image, label


@dataclass
class LabeledBBox:
    class_index: int
    xn: float
    yn: float
    wn: float
    hn: float

    def to_yolov5_line(self):
        return f'{self.class_index} {self.xn} {self.yn} {self.wn} {self.hn}'

    @staticmethod
    def from_yolov5_line(line):
        class_index, x, y, w, h = line.split(' ')
        class_index = int(class_index)
        x, y, w, h = float(x), float(y), float(w), float(h)
        x -= w / 2
        y -= h / 2
        return LabeledBBox(class_index, x, y, w, h)

    @staticmethod
    def from_yolov5_file(file_path):
        return [LabeledBBox.from_yolov5_line(line) for line in read_lines(file_path)]
