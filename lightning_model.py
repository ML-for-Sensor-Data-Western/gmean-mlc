import pytorch_lightning as pl
import torch
from torchvision import models as torch_models

from torchvision.models.vision_transformer import ViT_B_16_Weights

import ml_models
import sewer_models

LR_STEPS = [30,60,80]

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
        lr_steps: list | None = None,
        criterion=torch.nn.BCEWithLogitsLoss,
        dropout=0.2,
        attention_dropout=0.1,
        **kwargs,
    ):
        super(MultiLabelModel, self).__init__()
        self.save_hyperparameters()

        self.num_classes = num_classes

        if model in MultiLabelModel.TORCHVISION_MODEL_NAMES:
            # self.model = torch_models.__dict__[model](num_classes=self.num_classes, weights=ViT_B_16_Weights.IMAGENET1K_V1.value)
            self.model = torch_models.__dict__[model](num_classes=self.num_classes, dropout=dropout, attention_dropout=attention_dropout)
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
        
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.lr_steps = lr_steps
        if lr_steps is None:
            self.lr_steps = LR_STEPS

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
            lr=self.learning_rate,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            # Always adjust when changing number of epochs
            # Sewer-ML paper recommends 1/3, 2/3, 8/9 of total epochs
            optim,
            milestones=self.lr_steps,
            gamma=0.1,
        )

        return [optim], [scheduler]
