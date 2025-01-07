"""
Tuning selected parameters using Optuna.

The parameters in GLOBAL_CONFIG are the total hyperparameters that can be tuned.
The same parameters are in terminal args as well.
Make sure the parameters which do not need to be tuned are properly defined in
terminal args and the parameters to be tuned (must be a subset of GLOBAL_CONFIG)
are stated in args.params.
"""

import os
from argparse import ArgumentParser
from functools import partial

import lightning.pytorch as pl
import optuna
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from optuna.integration import PyTorchLightningPruningCallback
from torchvision import transforms

from lightning_datamodules import MultiLabelDataModule
from lightning_model import MultiLabelModel
from loss import HybridLoss


def objective(trial: optuna.trial.Trial, args):
    # pl.seed_everything(1234567890)

    params = {
        "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256])
        if "batch_size" in args.params
        else args.batch_size,
        "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.1)
        if "learning_rate" in args.params
        else args.learning_rate,
        "lr_decay": trial.suggest_float("lr_decay", 0.01, 0.1)
        if "lr_decay" in args.params
        else args.lr_decay,
        "momentum": trial.suggest_float("momentum", 0.5, 0.9)
        if "momentum" in args.params
        else args.momentum,
        "weight_decay": trial.suggest_float("weight_decay", 0.0001, 0.001)
        if "weight_decay" in args.params
        else args.weight_decay,
        "class_balancing_beta": trial.suggest_categorical(
            "class_balancing_beta", [0.9, 0.99, 0.999, 0.9999]
        )
        if "class_balancing_beta" in args.params
        else args.class_balancing_beta,
        "meta_loss_weight": trial.suggest_float("meta_loss_weight", 0.0, 1.0)
        if "meta_loss_weight" in args.params
        else args.meta_loss_weight,
        "meta_loss_beta": trial.suggest_float("meta_loss_beta", 0.0, 1.0)
        if "meta_loss_beta" in args.params
        else args.meta_loss_beta,
    }

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

    dm = MultiLabelDataModule(
        batch_size=params["batch_size"],
        workers=args.workers,
        ann_root=args.ann_root,
        data_root=args.data_root,
        train_transform=train_transform,
        eval_transform=eval_transform,
        only_defects=False,
    )

    dm.setup("fit")

    criterion = HybridLoss(
        class_counts=dm.class_counts,
        normal_count=dm.num_train_samples - dm.defect_count,
        class_balancing_beta=params["class_balancing_beta"],
        base_loss=args.base_loss,
        focal_gamma=args.focal_gamma,
        meta_loss_weight=params["meta_loss_weight"],
        meta_loss_beta=params["meta_loss_beta"],
    )

    light_model = MultiLabelModel(
        model=args.model,
        num_classes=dm.num_classes,
        criterion=criterion,
        learning_rate=params["learning_rate"],
        momentum=params["momentum"],
        weight_decay=params["weight_decay"],
        batch_size=params["batch_size"],
        lr_steps=args.lr_steps,
    )

    logger = TensorBoardLogger(
        save_dir=args.log_save_dir,
        name=args.model,
        version="e2e-version_" + str(args.log_version),
    )

    logger_path = os.path.join(
        args.log_save_dir, args.model, "e2e-version_" + str(args.log_version)
    )

    tune_callback = PyTorchLightningPruningCallback(trial, monitor=args.metric)
    ckpt_callback = ModelCheckpoint(
        dirpath=logger_path,
        filename=f"trial{trial.number:03}-" + "{epoch:03d}" + f"-{args.metric}",
        save_top_k=1,
        save_last=False,
        verbose=False,
        monitor=args.metric,
        mode="max" if args.metric in ["val_ap", "val_f1", "val_f2"] else "min",
    )

    trainer = pl.Trainer(
        devices=args.gpus,
        precision=args.precision,
        max_epochs=args.max_epochs,
        benchmark=True,
        logger=logger,
        callbacks=[tune_callback, ckpt_callback],
        enable_progress_bar=False,
    )

    torch.set_float32_matmul_precision(args.matmul_precision)
    trainer.fit(light_model, dm)

    return ckpt_callback.best_model_score


def tune_parameters(args):
    sampler = optuna.samplers.TPESampler(n_startup_trials=5, seed=12345)
    pruner = (
        optuna.pruners.MedianPruner(
            n_startup_trials=5, n_warmup_steps=32, interval_steps=2, n_min_trials=5
        )
        if args.pruning
        else optuna.pruners.NopPruner()
    )

    metric_optim_mode = (
        "maximize" if args.metric in ["val_ap", "val_f1", "val_f2"] else "minimize"
    )
    study = optuna.create_study(
        study_name=args.model + "_e2e_version_" + str(args.log_version),
        storage="sqlite:///gmean_mlc.db",
        sampler=sampler,
        pruner=pruner,
        direction=metric_optim_mode,
        load_if_exists=True,
    )

    partial_objective = partial(objective, args=args)

    study.optimize(
        partial_objective,
        n_trials=args.num_trials,
        n_jobs=args.max_concurrent_trials,
        show_progress_bar=True,
    )

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--ann_root", type=str, default="./annotations")
    parser.add_argument("--data_root", type=str, default="./Data")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--gpus", nargs="+", type=int, default=[0])
    parser.add_argument("--num_trials", type=int, default=50)
    parser.add_argument("--max_concurrent_trials", type=int, default=1)
    parser.add_argument("--pruning", action="store_true")
    parser.add_argument(
        "--precision", type=str, default="16-mixed", choices=["16-mixed", "32"]
    )
    parser.add_argument(
        "--matmul_precision",
        type=str,
        default="medium",
        choices=["medium", "high", "highest"],
    )
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
    parser.add_argument(
        "--metric",
        type=str,
        help="metric to optimizer",
        choices=["val_ap", "val_f1", "val_f2"],
    )
    parser.add_argument(
        "--params",
        nargs="+",
        type=str,
        help="params to optimize",
        default=["meta_loss_weight", "meta_loss_beta", "class_balancing_beta"],
    )
    parser.add_argument("--log_save_dir", type=str, default="./logs")
    parser.add_argument("--log_version", type=int, default=1)

    args = parser.parse_args()

    # Adjust learning rate to amount of GPUs
    # args.workers = max(0, min(8, 4 * len(args.gpus_per_trial)))
    # args.learning_rate = args.learning_rate * (len(args.gpus_per_trial) * args.batch_size) / 256

    print("Arguments: ", args)

    tune_parameters(args)
