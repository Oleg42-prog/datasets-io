import os
import yaml


def load_yaml(file_path):
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    return data


def load_text(file_path):
    with open(file_path, 'r') as file:
        data = file.read()
    return data


def read_lines(file_path):
    with open(file_path, 'r') as file:
        return file.readlines()


def change_file_name_extension(file_path, new_extension):
    dir_name, file_name = os.path.split(file_path)
    file_title, _ = os.path.splitext(file_name)
    if new_extension.startswith('.'):
        new_extension = new_extension[1:]
    new_file_name = f'{file_title}.{new_extension}'
    return os.path.join(dir_name, new_file_name)
