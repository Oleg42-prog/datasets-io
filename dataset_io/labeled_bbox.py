from dataclasses import dataclass
from utils import read_lines


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
