import pytorch_lightning as pl
import torch
from torchvision import models as torch_models

from eval_metrics import CustomMultiLabelAveragePrecision, MaxMultiLabelFbetaScore

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
            
        self.biases = torch.nn.Parameter(torch.zeros(1, 17), requires_grad=True)

        self.criterion = criterion
        self.max_f1 = MaxMultiLabelFbetaScore(num_labels=self.num_classes, beta=1.)
        self.max_f2 = MaxMultiLabelFbetaScore(num_labels=self.num_classes, beta=2.)
        self.ap = CustomMultiLabelAveragePrecision(num_labels=self.num_classes)

        if callable(getattr(self.criterion, "set_device", None)):
            self.criterion.set_device(self.device)
        
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.lr_steps = lr_steps
        if lr_steps is None:
            self.lr_steps = LR_STEPS
        self.batch_size = batch_size

    def forward(self, x):
        logits_before_bias = self.model(x)
        logits = logits_before_bias + self.biases
        return logits_before_bias, logits

    def loss(self, logits_before_bias, logits, y):
        y = y.float()
        loss = self.criterion(logits_before_bias, logits, y)

        return loss
    
    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        logits_before_bias, logits = self(x)

        loss = self.loss(logits_before_bias, logits, y)
        
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
        logits_before_bias, logits = self(x)
        loss = self.loss(logits_before_bias, logits, y)
        
        self.max_f1.update(logits, y)
        self.max_f2.update(logits, y)
        self.ap.update(logits, y)
        
        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=self.batch_size,
        )
        return loss
    
    def on_validation_epoch_end(self) -> None:
        self.log("val_ap", self.ap.compute(), on_step=False, on_epoch=True, prog_bar=True)
        self.ap.reset()    
        self.log("val_max_f1", self.max_f1.compute(), on_step=False, on_epoch=True, prog_bar=True)
        self.max_f1.reset()
        self.log("val_max_f2", self.max_f2.compute(), on_step=False, on_epoch=True, prog_bar=True)
        self.max_f2.reset()
    
    def test_step(self, batch, batch_idx):
        x, y, _ = batch
        logits_before_bias, logits = self(x)
        loss = self.loss(logits_before_bias, logits, y)
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
