import albumentations as A
import cv2

from albumentations.pytorch import ToTensorV2
from config import IMAGE_SIZE


class Albumentation:
    def __init__(self):
        self.transform = A.Compose(
            [
                A.LongestMaxSize(max_size=IMAGE_SIZE),
                A.PadIfNeeded(
                    min_height=IMAGE_SIZE,
                    min_width=IMAGE_SIZE,
                    border_mode=cv2.BORDER_CONSTANT,
                ),
                A.Normalize(
                    mean=[0.5544, 0.4927, 0.4470],
                    std=[0.2617, 0.2530, 0.2621],
                    max_pixel_value=255,
                ),
                ToTensorV2(),
            ],
            bbox_params=A.BboxParams(
                format="yolo", min_visibility=0.4, label_fields=[]
            ),
        )

    def __call__(self, image, bboxes):
        new = self.transform(image=image, bboxes=bboxes)
        return new["image"], new["bboxes"]
