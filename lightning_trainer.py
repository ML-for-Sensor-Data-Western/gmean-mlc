import os
from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from torchvision import transforms

from lightning_datamodules import (
    BinaryDataModule,
    BinaryRelevanceDataModule,
    MultiLabelDataModule,
)
from lightning_model import MultiLabelModel


class CustomLogger(TensorBoardLogger):
    def log_metrics(self, metrics, step=None):
        if "epoch" in metrics:
            step = metrics["epoch"]
        super().log_metrics(metrics, step)


def main(args):
    pl.seed_everything(1234567890)

    # Init data with transforms
    img_size = 299 if args.model in ["inception_v3", "chen2018_multilabel"] else 224

    train_transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.523, 0.453, 0.345], std=[0.210, 0.199, 0.154]),
        ]
    )

    eval_transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.523, 0.453, 0.345], std=[0.210, 0.199, 0.154]),
        ]
    )

    if args.training_mode == "e2e":
        dm = MultiLabelDataModule(
            batch_size=args.batch_size,
            workers=args.workers,
            ann_root=args.ann_root,
            data_root=args.data_root,
            train_transform=train_transform,
            eval_transform=eval_transform,
            only_defects=False,
        )
    elif args.training_mode == "defect":
        dm = MultiLabelDataModule(
            batch_size=args.batch_size,
            workers=args.workers,
            ann_root=args.ann_root,
            data_root=args.data_root,
            train_transform=train_transform,
            eval_transform=eval_transform,
            only_defects=True,
        )
    elif args.training_mode == "binary":
        dm = BinaryDataModule(
            batch_size=args.batch_size,
            workers=args.workers,
            ann_root=args.ann_root,
            data_root=args.data_root,
            train_transform=train_transform,
            eval_transform=eval_transform,
        )
    elif args.training_mode == "binaryrelevance":
        assert (
            args.br_defect is not None
        ), "Training mode is 'binary_relevance', but no 'br_defect' was stated"
        dm = BinaryRelevanceDataModule(
            batch_size=args.batch_size,
            workers=args.workers,
            ann_root=args.ann_root,
            data_root=args.data_root,
            train_transform=train_transform,
            eval_transform=eval_transform,
            defect=args.br_defect,
        )
    else:
        raise Exception("Invalid training_mode '{}'".format(args.training_mode))

    dm.prepare_data()
    dm.setup("fit")

    # Init our model
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=dm.class_weights)

    light_model = MultiLabelModel(
        num_classes=dm.num_classes,
        criterion=criterion,
        lr_steps=[30, 60, 80],
        **vars(args),
    )

    # train
    prefix = "{}-".format(args.training_mode)
    if args.training_mode == "binaryrelevance":
        prefix += args.br_defect
    """
    logger = TensorBoardLogger(
        save_dir=args.log_save_dir,
        name=args.model,
        version=prefix + "version_" + str(args.log_version),
    )
    """
    logger = CustomLogger(
        save_dir=args.log_save_dir,
        name=args.model,
        version=prefix + "version_" + str(args.log_version),
    )

    logger_path = os.path.join(
        args.log_save_dir, args.model, prefix + "version_" + str(args.log_version)
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=logger_path,
        filename="{epoch:02d}-{val_loss:.2f}",
        save_top_k=3,
        save_last=True,
        verbose=False,
        monitor="val_loss",
        mode="min",
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")

    early_stopper = EarlyStopping(monitor="val_loss", mode="min", patience=10)

    trainer = pl.Trainer(
        num_nodes=args.gpus,
        precision=args.precision,
        max_epochs=args.max_epochs,
        benchmark=True,
        logger=logger,
        callbacks=[checkpoint_callback, lr_monitor, early_stopper],
    )

    try:
        trainer.fit(light_model, dm)
    except Exception as e:
        print(e)
        with open(os.path.join(logger_path, "error.txt"), "w") as f:
            f.write(str(e))


def run_cli():
    # add PROGRAM level args
    parser = ArgumentParser()
    parser.add_argument("--ann_root", type=str, default="./annotations")
    parser.add_argument("--data_root", type=str, default="./Data")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--log_save_dir", type=str, default="./logs")
    parser.add_argument("--log_version", type=int, default=1)
    parser.add_argument(
        "--training_mode",
        type=str,
        default="e2e",
        choices=["e2e", "binary", "binaryrelevance", "defect"],
    )
    parser.add_argument(
        "--br_defect",
        type=str,
        default=None,
        choices=[
            None,
            "RB",
            "OB",
            "PF",
            "DE",
            "FS",
            "IS",
            "RO",
            "IN",
            "AF",
            "BE",
            "FO",
            "GR",
            "PH",
            "PB",
            "OS",
            "OP",
            "OK",
        ],
    )
    # Trainer args
    parser.add_argument("--precision", type=int, default=32, choices=[16, 32])
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--gpus", type=int, default=1)
    # Data args
    parser.add_argument(
        "--batch_size", type=int, default=64, help="Size of the batch per GPU"
    )
    # Model args
    parser.add_argument(
        "--model", type=str, default="resnet18", choices=MultiLabelModel.MODEL_NAMES
    )
    parser.add_argument("--learning_rate", type=float, default=1e-2)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=0.0001)

    args = parser.parse_args()

    # Adjust learning rate to amount of GPUs
    args.workers = max(0, min(8, 4 * args.gpus))
    args.learning_rate = args.learning_rate * (args.gpus * args.batch_size) / 256

    main(args)


if __name__ == "__main__":
    run_cli()
