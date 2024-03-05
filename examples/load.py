from PIL import ImageDraw
from dataset_io.dataset import Dataset, DatasetPart


def convert_bbox_xywhn_to_xywh(bbox, original_image_size):
    x, y, w, h = bbox.xn, bbox.yn, bbox.wn, bbox.hn

    x *= original_image_size[0]
    y *= original_image_size[1]
    w *= original_image_size[0]
    h *= original_image_size[1]

    return x, y, w, h


def draw_xywhn_bbox(image, bboxes):
    draw = ImageDraw.Draw(image)
    for bbox in bboxes:
        x, y, w, h = convert_bbox_xywhn_to_xywh(bbox, image.size)
        draw.rectangle([x, y, x + w, y + h], outline='red')
    return image


dataset = Dataset('data')
for image_name, image, labels in dataset.lazy_iterate(DatasetPart.VAL):
    new_image = draw_xywhn_bbox(image, labels)
    image.save(f'out/{image_name}.png')
