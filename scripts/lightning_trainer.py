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
import wandb
from lightning.pytorch.loggers import WandbLogger
from torchvision import transforms

from gmean_mlc.lightning_datamodules import MultiLabelDataModule
from gmean_mlc.lightning_model import MultiLabelModel
from gmean_mlc.loss import HybridLoss

WANDB_PROJECT_NAME = "gmean-mlc"

SEWER_MEAN = [0.523, 0.453, 0.345]
SEWER_STD = [0.210, 0.199, 0.154]

COCO_MEAN = [0.470, 0.447, 0.408]
COCO_STD = [0.233, 0.228, 0.231]

# To be checked
CHEST_MEAN = [0.506, 0.506, 0.506]
CHEST_STD = [0.230, 0.230, 0.230]

class CustomLogger(WandbLogger):
    def log_metrics(self, metrics, step=None):
        if "epoch" in metrics:
            step = metrics["epoch"]
        super().log_metrics(metrics, step)


def main(args):
    pl.seed_everything(1234567890)

    # Init data with transforms
    img_size = 299 if args.model in ["inception_v3", "chen2018_multilabel"] else 224
    
    if args.dataset == "sewer":
        data_mean = SEWER_MEAN
        data_std = SEWER_STD
    elif args.dataset == "coco":
        data_mean = COCO_MEAN
        data_std = COCO_STD
    elif args.dataset == "chest":
        data_mean = CHEST_MEAN
        data_std = CHEST_STD
    else:
        raise Exception("Invalid dataset '{}'".format(args.dataset))
    

    train_transform_list = [
        transforms.Resize((img_size, img_size)),
    ]

    eval_transform_list = [
        transforms.Resize((img_size, img_size)),
    ]

    if args.dataset == "chest":
        train_transform_list.append(transforms.Grayscale(num_output_channels=3))
        eval_transform_list.append(transforms.Grayscale(num_output_channels=3))

    train_transform_list += [
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=5),
        transforms.ToTensor(),
        transforms.Normalize(mean=data_mean, std=data_std),
    ]

    eval_transform_list += [
        transforms.ToTensor(),
        transforms.Normalize(mean=data_mean, std=data_std),
    ]

    train_transform = transforms.Compose(train_transform_list)
    eval_transform = transforms.Compose(eval_transform_list)


    dm = MultiLabelDataModule(
        dataset=args.dataset,
        batch_size=args.batch_size,
        workers=args.workers,
        ann_root=args.ann_root,
        data_root=args.data_root,
        train_transform=train_transform,
        eval_transform=eval_transform,
        only_defects=False,
    )

    dm.prepare_data()
    dm.setup("fit")

    # Init model
    criterion = HybridLoss(
        class_counts=dm.class_counts,
        normal_count=dm.num_train_samples - dm.any_class_count,
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

    logger_path = os.path.join(args.log_save_dir, "version_" + str(args.log_version))

    if not os.path.exists(logger_path):
        os.makedirs(logger_path)

    logger = CustomLogger(
        name="version_" + str(args.log_version),
        id = wandb.util.generate_id(),
        save_dir=logger_path,
        version=str(args.log_version),
        project=args.wandb_project,
        log_model=True,
        entity="gmean-mlc", ############
    )

    logger.experiment.config.update(vars(args), allow_val_change=True)

    for metric, direction in zip(
        ["val_ap", "val_f1", "val_f2", "val_bce", "val_loss"],
        ["max", "max", "max", "min", "min"],
    ):
        logger.experiment.define_metric(metric, summary=direction)

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
        gradient_clip_val=1.0,
    )

    try:
        torch.set_float32_matmul_precision(args.matmul_precision)
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
    parser.add_argument("--wandb_project", type=str, default=WANDB_PROJECT_NAME)
    parser.add_argument("--dataset", type=str, default="sewer", choices=["sewer", "coco", "chest"])
    parser.add_argument("--ann_root", type=str, default="./annotations")
    parser.add_argument("--data_root", type=str, default="./Data")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--gpus", nargs="+", type=int, default=[0])
    parser.add_argument("--checkpoint", type=str, default=None, help="If resuming")
    parser.add_argument("--log_save_dir", type=str, default="./logs")
    parser.add_argument("--log_version", type=str, default=1) ##########
    parser.add_argument(
        "--precision", type=str, default="32", choices=["16-mixed", "32"]
    )
    parser.add_argument(
        "--matmul_precision",
        type=str,
        default="highest",
        choices=["medium", "high", "highest"],
    )
    parser.add_argument("--max_epochs", type=int, default=40)
    parser.add_argument("--optimizer_type", type=str, default="sgd", choices=["sgd", "adamW"])
    parser.add_argument("--learning_rate", type=float, default=0.1)
    parser.add_argument("--min_lr", type=float, default=1e-5)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--lr_steps", nargs="+", type=int, default=[20, 30])
    parser.add_argument("--lr_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=2000)
    parser.add_argument("--warmup_start_factor", type=float, default=0.01)
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
