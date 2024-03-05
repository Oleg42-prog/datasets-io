import yaml


def load_yaml(file_path):
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    return data


def load_text(file_path):
    with open(file_path, 'r') as file:
        data = file.read()
    return data
