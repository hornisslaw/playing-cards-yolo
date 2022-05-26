import torch
import torch.nn as nn

from config import DEVICE
from utils import intersection_over_union


class YoloLoss(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.entropy = nn.CrossEntropyLoss()
        self.sigmoid = nn.Sigmoid()

        # Constants signifying how much to pay for each respective part of the loss
        self.lambda_class = 1
        self.lambda_noobj = 10
        self.lambda_obj = 1
        self.lambda_box = 10
        self.scaled_anchors = module.scaled_anchors

    def forward(self, predictions, targets):
        anchors = self.scaled_anchors
        loss = []
        for i, (prediction, target) in enumerate(zip(predictions, targets)):
            target = target.to(DEVICE)
            # Check where obj and noobj (we ignore if target == -1)
            obj = target[..., 0] == 1
            noobj = target[..., 0] == 0

            no_object_loss = self.bce(
                (prediction[..., 0:1][noobj]),
                (target[..., 0:1][noobj]),
            )

            anchor = anchors[i].reshape(1, 3, 1, 1, 2)
            box_preds = torch.cat(
                [
                    self.sigmoid(prediction[..., 1:3]),
                    torch.exp(prediction[..., 3:5]) * anchor,
                ],
                dim=-1,
            )
            ious = intersection_over_union(
                box_preds[obj], target[..., 1:5][obj]
            ).detach()
            object_loss = self.mse(
                self.sigmoid(prediction[..., 0:1][obj]), ious * target[..., 0:1][obj]
            )

            prediction[..., 1:3] = self.sigmoid(prediction[..., 1:3])  # x,y coordinates
            target[..., 3:5] = torch.log(
                (1e-16 + target[..., 3:5] / anchor)
            )  # width, height coordinates
            box_loss = self.mse(prediction[..., 1:5][obj], target[..., 1:5][obj])

            class_loss = self.entropy(
                (prediction[..., 5:][obj]),
                (target[..., 5][obj].long()),
            )

            loss.append(
                self.lambda_box * box_loss
                + self.lambda_obj * object_loss
                + self.lambda_noobj * no_object_loss
                + self.lambda_class * class_loss
            )

        return sum(loss)
