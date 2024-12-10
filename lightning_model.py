import pytorch_lightning as pl
import torch
from torchvision import models as torch_models

from torchmetrics.classification import MultilabelF1Score, MultilabelFBetaScore
from eval_metrics import CustomMultiLabelAveragePrecision

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
        batch_size=64,
        lr_steps: list | None = None,
        lr_decay: float = 0.1,
        criterion=torch.nn.BCEWithLogitsLoss,
        dropout=0.2,
        attention_dropout=0.1,
        **kwargs,
    ):
        super(MultiLabelModel, self).__init__()
        self.save_hyperparameters(ignore="criterion")

        self.num_classes = num_classes

        if model in MultiLabelModel.TORCHVISION_MODEL_NAMES:
            self.model = torch_models.__dict__[model](num_classes=self.num_classes, dropout=dropout, attention_dropout=attention_dropout)
        elif model in MultiLabelModel.SEWER_MODEL_NAMES:
            self.model = sewer_models.__dict__[model](num_classes=self.num_classes)
        elif model in MultiLabelModel.MULTILABEL_MODEL_NAMES:
            self.model = ml_models.__dict__[model](num_classes=self.num_classes)
        else:
            raise ValueError(
                "Got model {}, but no such model is in this codebase".format(model)
            )            

        self.criterion = criterion
        self.bce = torch.nn.BCEWithLogitsLoss()
        self.ap = CustomMultiLabelAveragePrecision(num_labels=self.num_classes)
        self.f1 = MultilabelF1Score(num_labels=self.num_classes, average="macro")
        self.f2 = MultilabelFBetaScore(num_labels=self.num_classes, beta=2., average="macro")

        if callable(getattr(self.criterion, "set_device", None)):
            self.criterion.set_device(self.device)
        
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.lr_steps = lr_steps
        if lr_steps is None:
            self.lr_steps = LR_STEPS
        self.lr_decay = lr_decay
        self.batch_size = batch_size

    def forward(self, x):
        logits = self.model(x)
        return logits
    
    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        logits = self(x)
        
        loss = self.criterion(logits, y.float())
        
        self.log(
            "train_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=self.batch_size,
        )
        
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, _ = batch
        logits = self(x)
        
        loss = self.criterion(logits, y.float())
        bce = self.bce(logits, y.float())
        
        self.ap.update(logits, y)
        self.f1.update(logits, y)
        self.f2.update(logits, y)
        
        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=self.batch_size,
        )
        self.log(
            "val_bce",
            bce,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=self.batch_size,
        )
        
        return loss
    
    def on_validation_epoch_end(self) -> None:
        self.log("val_ap", self.ap.compute(), on_step=False, on_epoch=True, prog_bar=True)
        self.ap.reset()
        self.log("val_f1", self.f1.compute(), on_step=False, on_epoch=True, prog_bar=True)
        self.f1.reset()
        self.log("val_f2", self.f2.compute(), on_step=False, on_epoch=True, prog_bar=True)
        self.f2.reset()
    
    def test_step(self, batch, batch_idx):
        x, y, _ = batch
        logits = self(x)
        loss = self.criterion(logits, y.float())
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
            optim,
            milestones=self.lr_steps,
            gamma=self.lr_decay,
        )

        return [optim], [scheduler]
