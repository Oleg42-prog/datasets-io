import os
from dataclasses import asdict

from PIL import Image

from utils import load_yaml, change_file_name_extension
from dataset_io.dataset_config import DatasetConfig, DatasetPart
from dataset_io.labeled_bbox import LabeledBBox


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
            image_file_name = os.path.basename(image_file_path)
            image = Image.open(image_file_path)
            label = LabeledBBox.from_yolov5_file(label_file_path)
            yield image_file_name, image, label
