import pathlib as p
import numpy as np
import torch

from PIL import Image, ImageFile
from torch.utils.data import Dataset, DataLoader

from augmentation import Albumentation
from config import SCALES, ANCHORS
from utils import iou_width_height, cells_to_bboxes, non_max_suppression, plot_image

ImageFile.LOAD_TRUNCATED_IMAGES = True


class YOLODataset(Dataset):
    def __init__(
        self,
        anchors,
        transform=None,
        dataset_name="",
        scale_sizes=SCALES,
        C=20,
    ):
        self.scale_sizes = scale_sizes
        self.C = C
        self.dataset_name = dataset_name
        self.image_list = self.load_images()
        self.label_list = self.load_labels()
        self.transform = transform
        self.anchors = torch.Tensor(sum(anchors, []))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        img_path = self.image_list[index]
        label_path = self.label_list[index]
        bboxes = np.roll(
            np.loadtxt(fname=label_path, delimiter=" ", ndmin=2), 4, axis=1
        ).tolist()
        image = np.array(Image.open(img_path).convert("RGB"))
        image, bboxes = self.transform(image=image, bboxes=bboxes)
        targets = self.build_targets(bboxes)

        return image, targets

    def load_images(self):
        images_path = p.Path(self.dataset_name) / "images"
        return sorted(list(images_path.glob("*.jpg")))

    def load_labels(self):
        labels_path = p.Path(self.dataset_name) / "labels"
        return sorted(list(labels_path.glob("*.txt")))

    def build_targets(self, bboxes):
        # TODO: divide this function into smaller functions for better readability
        self.num_anchors = self.anchors.shape[0]
        self.num_anchors_per_scale = self.num_anchors // 3
        self.ignore_iou_thresh = 0.5

        targets = [
            torch.zeros((self.num_anchors // 3, S, S, 6)) for S in self.scale_sizes
        ]  # [is_there_object, x, y, w, h, class]
        for box in bboxes:
            iou_anchors = iou_width_height(torch.tensor(box[2:4]), self.anchors)
            anchor_indices = iou_anchors.argsort(descending=True, dim=0)
            x, y, width, height, class_label = box
            has_anchor = [False] * 3  # each scale should have one anchor
            for anchor_idx in anchor_indices:
                scale_idx = anchor_idx // self.num_anchors_per_scale
                anchor_on_scale = anchor_idx % self.num_anchors_per_scale
                S = self.scale_sizes[scale_idx]
                i, j = int(S * y), int(S * x)  # which cell
                anchor_taken = targets[scale_idx][anchor_on_scale, i, j, 0]
                if not anchor_taken and not has_anchor[scale_idx]:
                    targets[scale_idx][anchor_on_scale, i, j, 0] = 1
                    x_cell, y_cell = S * x - j, S * y - i  # both between [0,1]
                    width_cell, height_cell = (
                        width * S,
                        height * S,
                    )  # can be greater than 1 since it's relative to cell
                    box_coordinates = torch.tensor(
                        [x_cell, y_cell, width_cell, height_cell]
                    )
                    targets[scale_idx][anchor_on_scale, i, j, 1:5] = box_coordinates
                    targets[scale_idx][anchor_on_scale, i, j, 5] = int(class_label)
                    has_anchor[scale_idx] = True

                elif (
                    not anchor_taken
                    and iou_anchors[anchor_idx] > self.ignore_iou_thresh
                ):
                    targets[scale_idx][
                        anchor_on_scale, i, j, 0
                    ] = -1  # ignore prediction
        return tuple(targets)


def test():
    test_dataset = YOLODataset(
                dataset_name="test",
                scale_sizes=SCALES,
                anchors=ANCHORS,
                transform=Albumentation())
    
    loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=True)

    S = [13, 26, 52]
    scaled_anchors = torch.tensor(ANCHORS) / (
        1 / torch.tensor(S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
    )
    for x, y in loader:
        boxes = []
        for i in range(y[0].shape[1]): # for i in range 3
            anchor = scaled_anchors[i]
            boxes += cells_to_bboxes(
                y[i], is_preds=False, S=y[i].shape[2], anchors=anchor
            )[0]
        boxes = non_max_suppression(boxes, iou_threshold=1, threshold=0.7, box_format="midpoint")
        # print(boxes)
        print(y[0].shape)
        print(x[0].shape)
        plot_image(x[0].permute(1, 2, 0).to("cpu"), boxes)
        break

if __name__ == "__main__":
    raise SystemExit(test())