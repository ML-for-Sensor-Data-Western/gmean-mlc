import os
from argparse import ArgumentParser
from typing import Optional

import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import TensorBoardLogger
from torchvision import transforms

from lightning_datamodules import (
    BinaryDataModule,
    MultiLabelDataModule,
)
from lightning_model import MultiLabelModel
from loss import HybridLoss


class CustomLogger(TensorBoardLogger):
    def log_metrics(self, metrics, step=None):
        if "epoch" in metrics:
            step = metrics["epoch"]
        super().log_metrics(metrics, step)


def main(args):
    # pl.seed_everything(1234567890)

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
    else:
        raise Exception("Invalid training_mode '{}'".format(args.training_mode))

    dm.prepare_data()
    dm.setup("fit")

    # Init model
    criterion = HybridLoss(
        class_counts=dm.class_counts,
        normal_count=dm.num_train_samples - dm.defect_count,
        class_balancing_beta=args.class_balancing_beta,
        base_loss=args.base_loss,
        focal_gamma=args.focal_gamma,
        meta_loss_weight=args.meta_loss_weight,
        meta_loss_beta=args.meta_loss_beta,
    )

    light_model = MultiLabelModel(
        num_classes=dm.num_classes,
        criterion=criterion,
        **vars(args),
    )
    

    # train
    prefix = "{}-".format(args.training_mode)

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
        save_top_k=1,
        save_last=True,
        verbose=False,
        monitor="val_loss",
        mode="min",
    )

    checkpoint_callback_ap = ModelCheckpoint(
        dirpath=logger_path,
        filename="{epoch:02d}-{val_ap:.2f}",
        save_top_k=1,
        save_last=False,
        verbose=False,
        monitor="val_ap",
        mode="max",
    )

    checkpoint_callback_f1 = ModelCheckpoint(
        dirpath=logger_path,
        filename="{epoch:02d}-{val_f1:.2f}",
        save_top_k=1,
        save_last=False,
        verbose=False,
        monitor="val_f1",
        mode="max",
    )

    checkpoint_callback_f2 = ModelCheckpoint(
        dirpath=logger_path,
        filename="{epoch:02d}-{val_f2:.2f}",
        save_top_k=1,
        save_last=False,
        verbose=False,
        monitor="val_f2",
        mode="max",
    )

    checkpoint_callback_bce = ModelCheckpoint(
        dirpath=logger_path,
        filename="{epoch:02d}-{val_bce:.2f}",
        save_top_k=1,
        save_last=False,
        verbose=False,
        monitor="val_bce",
        mode="min",
    )

    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    # early_stopper = EarlyStopping(monitor="val_ap", mode="max", patience=20)

    trainer = pl.Trainer(
        devices=args.gpus,
        num_nodes=1,
        precision=args.precision,
        max_epochs=args.max_epochs,
        benchmark=True,
        logger=logger,
        callbacks=[
            checkpoint_callback,
            checkpoint_callback_ap,
            checkpoint_callback_f1,
            checkpoint_callback_f2,
            checkpoint_callback_bce,
            lr_monitor,
            # early_stopper,
        ],
    )

    try:
        if args.checkpoint is not None:
            trainer.fit(light_model, dm, ckpt_path=args.checkpoint)
        else:
            trainer.fit(light_model, dm)

        best_val_ap = checkpoint_callback_ap.best_model_score
        best_val_f1 = checkpoint_callback_f1.best_model_score
        best_val_f2 = checkpoint_callback_f2.best_model_score
        best_val_bce = checkpoint_callback_bce.best_model_score
        best_val_loss = checkpoint_callback.best_model_score

        print(f"""
        Best val_ap: {best_val_ap:.4f}
        Best val_f1: {best_val_f1:.4f}
        Best val_f2: {best_val_f2:.4f}
        Best val_bce: {best_val_bce:.4f}
        Best val_loss: {best_val_loss:.4f}
        """)

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
    parser.add_argument("--gpus", nargs="+", type=int, default=[0])
    parser.add_argument("--checkpoint", type=str, default=None, help="If resuming")
    parser.add_argument("--log_save_dir", type=str, default="./logs")
    parser.add_argument("--log_version", type=int, default=1)
    parser.add_argument(
        "--training_mode",
        type=str,
        default="e2e",
        choices=["e2e", "binary", "defect"],
    )
    parser.add_argument("--precision", type=int, default=32, choices=[16, 32])
    parser.add_argument("--max_epochs", type=int, default=40)
    parser.add_argument("--learning_rate", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--lr_steps", nargs="+", type=int, default=[20, 30])
    parser.add_argument("--lr_decay", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument(
        "--model", type=str, default="resnet18", choices=MultiLabelModel.MODEL_NAMES
    )
    parser.add_argument("--mtl_heads", action="store_true")
    parser.add_argument(
        "--base_loss", type=str, default="focal", choices=["focal", "bce"]
    )
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    parser.add_argument("--class_balancing_beta", type=float, default=0.9999)
    parser.add_argument("--meta_loss_weight", type=float, default=0.1)
    parser.add_argument("--meta_loss_beta", type=float, default=0.1)

    args = parser.parse_args()
    print(args)

    # Adjust learning rate to amount of GPUs
    # args.workers = max(0, min(8, 4 * len(args.gpus)))
    # args.learning_rate = args.learning_rate * (len(args.gpus) * args.batch_size) / 256

    main(args)


if __name__ == "__main__":
    run_cli()
