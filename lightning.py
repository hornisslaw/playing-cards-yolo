import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import Dataset, DataLoader

from config import (
    NUM_CLASSES,
    NUM_WORKERS,
    PIN_MEMORY,
    SCALES,
    ANCHORS,
    BATCH_SIZE,
    LEARNING_RATE,
    WEIGHT_DECAY,
)
from dataset import YOLODataset
from augmentation import Albumentation
from metrics import check_class_accuracy

from utils import (
    cells_to_bboxes,
    non_max_suppression,
    plot_image,
    get_evaluation_bboxes,
)

from torchmetrics.detection.mean_ap import MeanAveragePrecision


class PlayingCardsDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size,
        train_dataset_path="train",
        test_dataset_path="test",
        val_dataset_path="valid",
        is_small_dataset = False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        num_classes=NUM_CLASSES,
    ):
        super().__init__()
        # self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.train_dataset_path = train_dataset_path
        self.test_dataset_path = test_dataset_path
        self.val_dataset_path = val_dataset_path
        self.is_small_dataset = is_small_dataset
        self.image_size = (416, 416)
        self.transform = Albumentation()
        self.num_classes = num_classes

    def prepare_data(self):
#         if not self.is_small_dataset:
#             # TODO: change to subprocess
#             # download full playing cards dataset
#             ! gdown 1COhcBiGMlSwqYY62oo8nbSxKMSYbHStC
#             ! unzip -q playing_cards.zip && rm playing_cards.zip
#         else:
#             # download mini playing cards dataset
#             ! gdown 1etYyC3Ws-GW01U0FEUAsAcuP273OwS7N
#             ! unzip -q playing_cards_small.zip && rm playing_cards_small.zip
        pass

    def setup(self, stage=None):
        # train/val
        if stage == "fit" or stage is None:
            self.train_dataset = YOLODataset(
                dataset_name=self.train_dataset_path,
                scale_sizes=SCALES,
                anchors=ANCHORS,
                transform=self.transform,
            )
            self.val_dataset = YOLODataset(
                dataset_name=self.val_dataset_path,
                scale_sizes=SCALES,
                anchors=ANCHORS,
                transform=self.transform,
            )
        # test
        if stage == "test" or stage is None:
            self.test_dataset = YOLODataset(
                dataset_name=self.test_dataset_path,
                scale_sizes=SCALES,
                anchors=ANCHORS,
                transform=self.transform,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
        )


class YoloV3LitModel(pl.LightningModule):
    def __init__(
        self,
        model,
        loss_model,
        batch_size=BATCH_SIZE,
        lr=LEARNING_RATE,
        num_classes=NUM_CLASSES,
    ):
        super().__init__()
        self.model = model
        self.loss_model = loss_model
        self.batch_size = batch_size
        self.lr = lr
        self.num_classes = num_classes
        self.current_epoch_training_loss = torch.tensor(0.0)

        self.metric = MeanAveragePrecision()

    def forward(self, x):
        return self.model(x)

    def compute_loss(self, x, y):
        return self.loss_model(x, y)

    def common_step(self, batch, batch_idx):
        x, y = batch
        outputs = self(x)
        loss = self.compute_loss(outputs, y)
        # display_image(x, outputs)

        return loss, outputs, y

    def on_step(self, batch, batch_idx):
        x, y = batch
        outputs = self(x)
        loss = self.compute_loss(outputs, y)
        acc = check_class_accuracy(outputs, y)
        pred_boxes, true_boxes = get_evaluation_bboxes(outputs, y, len(batch[0]))
        self.metric.update([pred_boxes], [true_boxes])
        result_map = self.metric.compute()
        return {"loss": loss, "map": result_map["map"]}

    def on_end(self, prefix, outs):
        avg_loss = torch.stack([o["loss"] for o in outs]).mean()
        # avg_class_acc = torch.stack([o["val_class_acc"] for o in outs]).mean()
        # avg_no_obj_acc = torch.stack([o["val_no_obj_acc"] for o in outs]).mean()
        # avg_obj_acc = torch.stack([o["val_obj_acc"] for o in outs]).mean()
        avg_map = torch.stack([o["map"] for o in outs]).mean()
        # pred = [o["pred_boxes"] for o in outs]
        # true = [o["true_boxes"] for o in outs]

        self.log(
            prefix + "_avg_loss", avg_loss, on_epoch=True, prog_bar=True, logger=True
        )
        self.log(
            prefix + "_avg_map", avg_map, on_epoch=True, prog_bar=True, logger=True
        )

    def training_epoch_end(self, outs):
        self.on_end("training", outs)

    def validation_epoch_end(self, outs):
        self.on_end("valid", outs)

    def test_epoch_end(self, outs):
        self.on_end("test", outs)

    def training_step(self, batch, batch_idx):
        return self.on_step(batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        return self.on_step(batch, batch_idx)

    def test_step(self, batch, batch_idx):
        return self.on_step(batch, batch_idx)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=WEIGHT_DECAY
        )
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=3, gamma=0.1
        )
        return [optimizer], [lr_scheduler]


def display_image(x, outputs):
    S = [13, 26, 52]
    scaled_anchors = torch.tensor(ANCHORS) / (
        1 / torch.tensor(S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
    )
    boxes = []
    for i in range(outputs[0].shape[1]):  # for i in range 3
        anchor = scaled_anchors[i]
        boxes += cells_to_bboxes(
            outputs[i], is_preds=True, S=outputs[i].shape[2], anchors=anchor
        )[0]
    boxes = non_max_suppression(
        boxes, iou_threshold=1, threshold=0.7, box_format="midpoint"
    )
    plot_image(x[0].permute(1, 2, 0).to("cpu"), boxes)


# class YoloV3LitModel(pl.LightningModule):
#     def __init__(
#         self,
#         model,
#         loss_model,
#         batch_size=BATCH_SIZE,
#         lr=LEARNING_RATE,
#         num_classes=NUM_CLASSES,
#     ):
#         super().__init__()
#         self.model = model
#         self.loss_model = loss_model
#         self.batch_size = batch_size
#         self.lr = lr
#         self.num_classes = num_classes
#         self.current_epoch_training_loss = torch.tensor(0.0)

#         self.metric = MeanAveragePrecision()

#     def forward(self, x):
#         return self.model(x)

#     def compute_loss(self, x, y):
#         return self.loss_model(x, y)

#     def common_step(self, batch, batch_idx):
#         x, y = batch
#         outputs = self(x)
#         loss = self.compute_loss(outputs, y)
#         # display_image(x, outputs)

#         return loss, outputs, y

#     def on_step(self, batch, batch_idx):
#         x, y = batch
#         outputs = self(x)
#         loss = self.compute_loss(outputs, y)
#         acc = check_class_accuracy(outputs, y)
#         pred_boxes, true_boxes = get_evaluation_bboxes(outputs, y, len(batch[0]))
#         self.metric.update([pred_boxes], [true_boxes])
#         result_map = self.metric.compute()
#         return {'loss': loss, 'map': result_map['map']}

#     def common_test_valid_step(self, batch, batch_idx):
#         loss, outputs, y = self.common_step(batch, batch_idx)
#         acc = check_class_accuracy(outputs, y)
#         pred_boxes, true_boxes = get_evaluation_bboxes(outputs, y, len(batch[0]))

#         return loss, acc, pred_boxes, true_boxes

#     def training_step(self, batch, batch_idx):
#         loss, _, _ = self.common_step(batch, batch_idx)
#         self.log(
#             "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
#         )
#         return {"loss": loss}

#     def training_epoch_end(self, outs):
#         self.current_epoch_training_loss = torch.stack([o["loss"] for o in outs]).mean()

#     def validation_step(self, batch, batch_idx):
#         # TODO: create function for step repetitive code
#         loss, acc, pred_boxes, true_boxes = self.common_test_valid_step(batch, batch_idx)
#         self.log(
#             "val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
#         )
#         self.log(
#             "val_class", acc[0], on_step=True, on_epoch=True, prog_bar=True, logger=True
#         )
#         self.log(
#             "val_no_obj",
#             acc[1],
#             on_step=True,
#             on_epoch=True,
#             prog_bar=True,
#             logger=True,
#         )
#         self.log(
#             "val_obj", acc[2], on_step=True, on_epoch=True, prog_bar=True, logger=True
#         )
#         # self.log('map', map, on_step=True, on_epoch=False, prog_bar=True, logger=True)
#         return {
#             "val_loss": loss,
#             "val_class_acc": acc[0],
#             "val_no_obj_acc": acc[1],
#             "val_obj_acc": acc[2],
#             "pred_boxes": pred_boxes,
#             "true_boxes": true_boxes
#         }

#     def validation_epoch_end(self, outs):
#         avg_loss = torch.stack([o["val_loss"] for o in outs]).mean()
#         avg_class_acc = torch.stack([o["val_class_acc"] for o in outs]).mean()
#         avg_no_obj_acc = torch.stack([o["val_no_obj_acc"] for o in outs]).mean()
#         avg_obj_acc = torch.stack([o["val_obj_acc"] for o in outs]).mean()

#         pred = [o["pred_boxes"] for o in outs]
#         true = [o["true_boxes"] for o in outs]
#         self.metric.update(pred, true)
#         result_map = self.metric.compute()
#         print(f"map: {result_map['map']}")
#         losses = {
#             "train": self.current_epoch_training_loss.item(),
#             "avg_class_acc": avg_class_acc.item(),
#             "val_no_obj_acc": avg_no_obj_acc.item(),
#             "val_obj_acc": avg_obj_acc.item(),
#             "result_map": result_map['map']
#         }
#         self.logger.experiment.add_scalars('train and vall losses', losses, self.current_epoch)

#     def test_step(self, batch, batch_idx):
#         loss, acc, pred_boxes, true_boxes = self.common_test_valid_step(batch, batch_idx)
#         self.log(
#             "test_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
#         )
#         self.log(
#             "test_class",
#             acc[0],
#             on_step=True,
#             on_epoch=True,
#             prog_bar=True,
#             logger=True,
#         )
#         self.log(
#             "test_no_obj",
#             acc[1],
#             on_step=True,
#             on_epoch=True,
#             prog_bar=True,
#             logger=True,
#         )
#         self.log(
#             "test_obj", acc[2], on_step=True, on_epoch=True, prog_bar=True, logger=True
#         )
#         self.metric.update([pred_boxes], [true_boxes])
#         result_map = self.metric.compute()
#         self.log('map', result_map['map'], on_step=True, on_epoch=True, prog_bar=True, logger=True)
#         return {
#             "test_loss": loss,
#             "test_class_acc": acc[0],
#             "test_no_obj_acc": acc[1],
#             "test_obj_acc": acc[2],
#             "test_pred_boxes": pred_boxes,
#             "test_true_boxes": true_boxes
#         }

#     def configure_optimizers(self):
#         optimizer = torch.optim.Adam(
#             self.parameters(), lr=self.lr, weight_decay=WEIGHT_DECAY
#         )
#         lr_scheduler = torch.optim.lr_scheduler.StepLR(
#             optimizer, step_size=3, gamma=0.1
#         )
#         return [optimizer], [lr_scheduler]
