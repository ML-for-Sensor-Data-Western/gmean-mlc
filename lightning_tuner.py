import os
from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch import nn
from torchvision import models as torch_models
from torchvision import transforms

import ml_models
import sewer_models
from lightning_datamodules import (
    BinaryDataModule,
    BinaryRelevanceDataModule,
    MultiLabelDataModule,
)

import ray
from ray import tune
from ray.tune.integration.pytorch_lightning import TuneReportCallback, TuneReportCheckpointCallback
from ray.tune.schedulers import ASHAScheduler

#had to redefine because of value erro: Expected a parent
class MyTuneReportCheckpointCallback(TuneReportCheckpointCallback, pl.Callback):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class CustomLogger(TensorBoardLogger):
    def log_metrics(self, metrics, step=None):
        if "epoch" in metrics:
            step = metrics['epoch']
        super().log_metrics(metrics, step)

class MultiLabelModel(pl.LightningModule):
    TORCHVISION_MODEL_NAMES = sorted(
        name
        for name in torch_models.__dict__
        if name.islower()
        and not name.startswith("__")
        and callable(torch_models.__dict__[name])
    )
    SEWER_MODEL_NAMES = sorted(
        name
        for name in sewer_models.__dict__
        if name.islower()
        and not name.startswith("__")
        and callable(sewer_models.__dict__[name])
    )
    MULTILABEL_MODEL_NAMES = sorted(
        name
        for name in ml_models.__dict__
        if name.islower()
        and not name.startswith("__")
        and callable(ml_models.__dict__[name])
    )
    MODEL_NAMES = TORCHVISION_MODEL_NAMES + SEWER_MODEL_NAMES + MULTILABEL_MODEL_NAMES

    def __init__(
        self,
        model="resnet18",
        num_classes=2,
        learning_rate=1e-2,
        momentum=0.9,
        weight_decay=0.0001,
        criterion=torch.nn.BCEWithLogitsLoss,
        **kwargs,
    ):
        super(MultiLabelModel, self).__init__()
        self.save_hyperparameters()

        self.num_classes = num_classes

        if model in MultiLabelModel.TORCHVISION_MODEL_NAMES:
            self.model = torch_models.__dict__[model](num_classes=self.num_classes)
        elif model in MultiLabelModel.SEWER_MODEL_NAMES:
            self.model = sewer_models.__dict__[model](num_classes=self.num_classes)
        elif model in MultiLabelModel.MULTILABEL_MODEL_NAMES:
            self.model = ml_models.__dict__[model](num_classes=self.num_classes)
        else:
            raise ValueError(
                "Got model {}, but no such model is in this codebase".format(model)
            )

        self.aux_logits = hasattr(self.model, "aux_logits")

        if self.aux_logits:
            self.train_function = self.aux_loss
        else:
            self.train_function = self.normal_loss
        self.criterion = criterion

        if callable(getattr(self.criterion, "set_device", None)):
            self.criterion.set_device(self.device)

    def forward(self, x):
        logits = self.model(x)
        return logits

    def aux_loss(self, x, y):
        y = y.float()
        y_hat, y_aux_hat = self(x)
        loss = self.criterion(y_hat, y) + 0.4 * self.criterion(y_aux_hat, y)

        return loss

    def normal_loss(self, x, y):
        y = y.float()
        y_hat = self(x)
        loss = self.criterion(y_hat, y)

        return loss

    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        loss = self.train_function(x, y)
        self.log(
            "train_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=self.hparams.batch_size,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, _ = batch
        loss = self.normal_loss(x, y)
        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=self.hparams.batch_size,
        )
        return loss

    def test_step(self, batch, batch_idx):
        x, y, _ = batch
        loss = self.normal_loss(x, y)
        self.log("test_loss", loss)
        return loss

    def configure_optimizers(self):
        optim = torch.optim.SGD(
            self.parameters(),
            lr=self.hparams.learning_rate,
            momentum=self.hparams.momentum,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            #Always adjust when changing number of epochs
            #Sewer-ML paper recommends 1/3, 2/3, 8/9 of total epochs
            optim, milestones=[30, 60, 80], gamma=0.1
        )

        return [optim], [scheduler]


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
        learning_rate=config["learning_rate"],
        momentum=config["momentum"],
        weight_decay=config["weight_decay"],
        batch_size=config["batch_size"]
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

    tune_callback = MyTuneReportCheckpointCallback(
        metrics={"val_loss": "val_loss"},
        on="validation_end"

    )

    lr_monitor = LearningRateMonitor(logging_interval="step")

    trainer = pl.Trainer(
        num_nodes=args.gpus,
        precision=args.precision,
        max_epochs=args.max_epochs,
        benchmark=True,
        logger=logger,
        callbacks=[checkpoint_callback, lr_monitor, tune_callback],
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
    parser.add_argument('--conda_env', type=str, default='Pytorch-Lightning')
    parser.add_argument('--notification_email', type=str, default='')
    parser.add_argument('--ann_root', type=str, default='./annotations')
    parser.add_argument('--data_root', type=str, default='./Data')
    parser.add_argument('--batch_size', type=int, default=64, help="Size of the batch per GPU")
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--log_save_dir', type=str, default="./logs")
    parser.add_argument('--log_version', type=int, default=1)
    parser.add_argument('--training_mode', type=str, default="e2e", choices=["e2e", "binary", "binaryrelevance", "defect"])
    parser.add_argument('--br_defect', type=str, default=None, choices=[None, "RB","OB","PF","DE","FS","IS","RO","IN","AF","BE","FO","GR","PH","PB","OS","OP","OK"])
    # Trainer args
    parser.add_argument('--precision', type=int, default=32, choices=[16, 32])
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--gpus', type=int, default=1)
    # Model args
    parser.add_argument('--model', type=str, default="resnet18", choices=MultiLabelModel.MODEL_NAMES)
    parser.add_argument('--learning_rate', type=float, default=1e-2)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=0.0001)

    parser.add_argument('--use_tuner', action='store_true', help='If true, Ray Tune will be implemented for hyperparameter')

    args = parser.parse_args()

    # Adjust learning rate to amount of GPUs
    args.workers = max(0, min(8, 4 * args.gpus))
    args.learning_rate = args.learning_rate * (args.gpus * args.batch_size) / 256

    config = {
        "batch_size": tune.choice([64, 128, 256, 512]),
        "learning_rate": tune.loguniform(1e-5, 1e-1),
        "momentum": tune.uniform(0.5, 0.9),
        "weight_decay": tune.loguniform(1e-6, 1e-3),
    }

    ashascheduler = ASHAScheduler(
        metric='val_loss',
        mode='min',
        max_t=args.max_epochs
    )

    if args.use_tuner:

        analysis = tune.run(
            tune.with_parameters(main, args=args),
            resources_per_trial={'cpu': 4, 'gpu': args.gpus},
            config=config,
            scheduler=ashascheduler,
            num_samples=10,
            name='tune_experiment',
            storage_path="E:\quinn\\ray_tune"
        )

        print('The best param values are: ', analysis.best_config)

    else:
        main(vars(args), args)


if __name__ == "__main__":
    run_cli()
