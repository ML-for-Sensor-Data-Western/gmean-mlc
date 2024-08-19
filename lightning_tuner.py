import os
from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from ray import tune
from ray.tune import CLIReporter
from ray.tune.integration.pytorch_lightning import TuneReportCheckpointCallback
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch
from torchvision import transforms

from lightning_datamodules import (
    BinaryDataModule,
    BinaryRelevanceDataModule,
    MultiLabelDataModule,
)
from lightning_model import MultiLabelModel


# had to redefine because of value erro: Expected a parent
class MyTuneReportCheckpointCallback(TuneReportCheckpointCallback, pl.Callback):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class CustomLogger(TensorBoardLogger):
    def log_metrics(self, metrics, step=None):
        if "epoch" in metrics:
            step = metrics["epoch"]
        super().log_metrics(metrics, step)


def main(config, args):
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
            batch_size=config["batch_size"],
            workers=args.workers,
            ann_root=args.ann_root,
            data_root=args.data_root,
            train_transform=train_transform,
            eval_transform=eval_transform,
            only_defects=False,
        )
    elif args.training_mode == "defect":
        dm = MultiLabelDataModule(
            batch_size=config["batch_size"],
            workers=args.workers,
            ann_root=args.ann_root,
            data_root=args.data_root,
            train_transform=train_transform,
            eval_transform=eval_transform,
            only_defects=True,
        )
    elif args.training_mode == "binary":
        dm = BinaryDataModule(
            batch_size=config["batch_size"],
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
            batch_size=config["batch_size"],
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
        model=args.model,
        num_classes=dm.num_classes,
        criterion=criterion,
        learning_rate=config["learning_rate"],
        momentum=config["momentum"],
        weight_decay=config["weight_decay"],
        batch_size=config["batch_size"],
        lr_steps=args.lr_steps,
        dropout=config["dropout"],
        attention_dropout=config["attention_dropout"],
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
        filename="{epoch:02d}-{val_acc:.2f}",
        save_top_k=3,
        save_last=True,
        verbose=False,
        monitor="val_acc",
        mode="max",
    )

    tune_callback = MyTuneReportCheckpointCallback(
        metrics={"val_acc": "val_acc"}, on="validation_end"
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")

    trainer = pl.Trainer(
        num_nodes=1,
        precision=args.precision,
        max_epochs=args.max_epochs,
        benchmark=True,
        logger=logger,
        callbacks=[checkpoint_callback, lr_monitor, tune_callback],
        enable_progress_bar=False,
    )

    try:
        trainer.fit(light_model, dm)
    except Exception as e:
        print("Error during trainer.fit: ", e)


def short_dirname(trial):
    return "trial_" + str(trial.trial_id)


def run_cli():
    # add PROGRAM level args
    parser = ArgumentParser()
    parser.add_argument("--ann_root", type=str, default="./annotations")
    parser.add_argument("--data_root", type=str, default="./Data")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--num_trials", type=int, default=50)
    parser.add_argument("--max_concurrent_trials", type=int, default=1)
    parser.add_argument("--cpus_per_trial", type=int, default=4)
    parser.add_argument("--gpus_per_trial", type=int, default=1)
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
    parser.add_argument("--max_epochs", type=int, default=50)
    parser.add_argument("--lr_steps", nargs="+", type=int, default=[15, 30, 40])
    # Model args
    parser.add_argument(
        "--model", type=str, default="resnet18", choices=MultiLabelModel.MODEL_NAMES
    )

    args = parser.parse_args()

    # Adjust learning rate to amount of GPUs
    # args.workers = max(0, min(8, 4 * len(args.gpus_per_trial)))
    # args.learning_rate = args.learning_rate * (len(args.gpus_per_trial) * args.batch_size) / 256

    config = {
        "batch_size": tune.choice([64, 128, 256]),
        "learning_rate": tune.choice([0.001, 0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1]),
        "momentum": tune.choice([0.5, 0.6, 0.7, 0.8, 0.9]),
        "weight_decay": tune.uniform(0.0001,0.1),
        "dropout": tune.uniform(0.1, 0.5),
        "attention_dropout": tune.uniform(0.05, 0.3),
    }
    
    search_algo = OptunaSearch(metric="val_acc", mode="max")

    search_scheduler = ASHAScheduler(
        time_attr="training_iteration", # default
        metric="val_acc",
        mode="max",
        max_t=100, # default
        grace_period=18,
        reduction_factor=4, # default
        brackets=1, # default
        stop_last_trials=True, # default
    )

    reporter = CLIReporter(
        parameter_columns=["batch_size", "learning_rate", "momentum", "weight_decay", "dropout", "attention_dropout"],
        metric_columns=["val_acc", "training_iteration"],
        print_intermediate_tables=False,
    )

    analysis = tune.run(
        tune.with_parameters(main, args=args),
        resources_per_trial={"cpu": args.cpus_per_trial, "gpu": args.gpus_per_trial},
        config=config,
        search_alg=search_algo,
        scheduler=search_scheduler,
        num_samples=args.num_trials,
        max_concurrent_trials=args.max_concurrent_trials,
        name="%s-version_%s" % (args.training_mode, args.log_version),
        storage_path="%s\%s" % (args.log_save_dir, args.model),
        progress_reporter=reporter,
        verbose=0,
        trial_dirname_creator=short_dirname,
    )

    print(
        "The best param values are: ",
        analysis.get_best_config(metric="val_acc", mode="max"),
    )


if __name__ == "__main__":
    run_cli()
