import os
from argparse import ArgumentParser

import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    EarlyStopping,
)
from pytorch_lightning.loggers import TensorBoardLogger
from torchvision import transforms

from lightning_datamodules import (
    BinaryDataModule,
    BinaryRelevanceDataModule,
    MultiLabelDataModule,
)
from lightning_model import MultiLabelModel
from loss import HybridLoss
from typing import Optional


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

    # Init model
    criterion = HybridLoss(
        class_counts=dm.class_counts,
        normal_count=dm.num_train_samples - dm.defect_count,
        beta=args.beta,
        base_loss=args.base_loss,
        focal_gamma=args.focal_gamma,
        meta_loss_weight=args.meta_loss_weight,
        push_mode=args.meta_push_mode,
    )

    light_model = MultiLabelModel(
        num_classes=dm.num_classes,
        criterion=criterion,
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

    checkpoint_callback_ap = ModelCheckpoint(
        dirpath=logger_path,
        filename="{epoch:02d}-{val_ap:.2f}",
        save_top_k=1,
        save_last=True,
        verbose=False,
        monitor="val_ap",
        mode="max",
    )

    checkpoint_callback_max_f1 = ModelCheckpoint(
        dirpath=logger_path,
        filename="{epoch:02d}-{val_max_f1:.2f}",
        save_top_k=1,
        save_last=False,
        verbose=False,
        monitor="val_max_f1",
        mode="max",
    )

    checkpoint_callback_max_f2 = ModelCheckpoint(
        dirpath=logger_path,
        filename="{epoch:02d}-{val_max_f2:.2f}",
        save_top_k=1,
        save_last=False,
        verbose=False,
        monitor="val_max_f2",
        mode="max",
    )
    
    checkpoint_callback_bce = ModelCheckpoint(
        dirpath=logger_path,
        filename="{epoch:02d}-{val_bce:.2f}",
        save_top_k=1,
        save_last=True,
        verbose=False,
        monitor="val_bce",
        mode="min",
    )
    
    checkpoint_callback_acc = ModelCheckpoint(
        dirpath=logger_path,
        filename="{epoch:02d}-{val_acc:.2f}",
        save_top_k=1,
        save_last=True,
        verbose=False,
        monitor="val_acc",
        mode="max",
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")

    # early_stopper = EarlyStopping(monitor="val_ap", mode="max", patience=20)

    trainer = pl.Trainer(
        devices=args.gpus,
        num_nodes=1,
        precision=args.precision,
        max_epochs=args.max_epochs,
        benchmark=True,
        logger=logger,
        callbacks=[
            checkpoint_callback_ap,
            checkpoint_callback_max_f1,
            checkpoint_callback_max_f2,
            checkpoint_callback_bce,
            checkpoint_callback_acc,
            lr_monitor,
            # early_stopper,
        ],
    )

    try:
        if args.checkpoint is not None:
            trainer.fit(light_model, dm, ckpt_path=args.checkpoint)
        else:
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
    parser.add_argument("--checkpoint", type=str, default=None, help="If resuming")
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
    parser.add_argument("--lr_steps", nargs="+", type=int, default=[30, 60, 80])
    parser.add_argument("--lr_decay", type=float, default=0.1)
    parser.add_argument("--gpus", nargs="+", type=int, default=[0])
    # Data args
    parser.add_argument(
        "--batch_size", type=int, default=64, help="Size of the batch per GPU"
    )
    # Model args
    parser.add_argument(
        "--model", type=str, default="resnet18", choices=MultiLabelModel.MODEL_NAMES
    )
    parser.add_argument(
        "--base_loss", type=str, default="focal", choices=["focal", "sigmoid"]
    )
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    parser.add_argument("--beta", type=float, default=0.9999)
    parser.add_argument("--meta_loss_weight", type=float, default=0.1)
    parser.add_argument(
        "--meta_push_mode",
        type=str,
        default="positive_push",
        choices=["positive_push", "all_push"],
    )
    parser.add_argument("--learning_rate", type=float, default=1e-1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--attention_dropout", type=float, default=0.1)

    args = parser.parse_args()
    print(args)

    # Adjust learning rate to amount of GPUs
    # args.workers = max(0, min(8, 4 * len(args.gpus)))
    # args.learning_rate = args.learning_rate * (len(args.gpus) * args.batch_size) / 256

    main(args)


if __name__ == "__main__":
    run_cli()
