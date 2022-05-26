import pytorch_lightning as pl

from model import YOLOv3
from loss import YoloLoss
from lightning import PlayingCardsDataModule, YoloV3LitModel
from config import BATCH_SIZE


def main():
    model = YOLOv3()
    loss_model = YoloLoss(model)
    dm = PlayingCardsDataModule(batch_size=BATCH_SIZE)
    yolov3 = YoloV3LitModel(model, loss_model)
    # wandb_logger = WandbLogger(project='wandb-lightning', job_type='train')
    trainer = pl.Trainer(
        check_val_every_n_epoch=2, num_sanity_val_steps=1, max_epochs=3
    )
    trainer.fit(model=yolov3, datamodule=dm)
    trainer.test(model=yolov3, datamodule=dm)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
