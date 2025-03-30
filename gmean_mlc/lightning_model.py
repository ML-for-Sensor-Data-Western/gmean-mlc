import lightning.pytorch as pl
import torch
from torchmetrics.classification import MultilabelF1Score, MultilabelFBetaScore
from torchvision import models as torch_models

from . import models as ml_models
from .metrics.val_metrics import CustomMultiLabelAveragePrecision

LR_STEPS = [30, 60, 80]


class MultiLabelModel(pl.LightningModule):
    TORCHVISION_MODEL_NAMES = sorted(
        name
        for name in torch_models.__dict__
        if name.islower()
        and not name.startswith("__")
        and callable(torch_models.__dict__[name])
    )
    MULTILABEL_MODEL_NAMES = sorted(
        name
        for name in ml_models.__dict__
        if name.islower()
        and not name.startswith("__")
        and callable(ml_models.__dict__[name])
    )
    MODEL_NAMES = TORCHVISION_MODEL_NAMES + MULTILABEL_MODEL_NAMES

    def __init__(
        self,
        model="resnet18",
        mtl_heads=False,
        num_classes=2,
        max_epochs: int = 40,
        optimizer_type="sgd",
        learning_rate=1e-1,
        min_lr=1e-5,
        momentum=0.9,
        weight_decay=0.0001,
        batch_size=256,
        lr_steps: list | None = None,
        lr_decay: float = 0.1,
        warmup_steps: int = 5,
        warmup_start_factor: float = 0.01,
        adam_beta1: float = 0.9,
        adam_beta2: float = 0.999,
        cosine_t0: int = 10,
        cosine_tmult: int = 2,
        criterion=torch.nn.BCEWithLogitsLoss,
        **kwargs,
    ):
        super(MultiLabelModel, self).__init__()
        self.save_hyperparameters(ignore="criterion")

        self.num_classes = num_classes
        self.max_epochs = max_epochs

        if model in MultiLabelModel.TORCHVISION_MODEL_NAMES:
            self.model = torch_models.__dict__[model](num_classes=self.num_classes)
        elif model in MultiLabelModel.MULTILABEL_MODEL_NAMES:
            self.model = ml_models.__dict__[model](num_classes=self.num_classes)
        else:
            raise ValueError(
                "Got model {}, but no such model is in this codebase".format(model)
            )

        self.optimizer_type = optimizer_type.lower()
        self.learning_rate = learning_rate
        self.min_lr = min_lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.lr_steps = lr_steps
        if lr_steps is None:
            self.lr_steps = LR_STEPS
        self.lr_decay = lr_decay
        self.warmup_steps = warmup_steps
        self.warmup_start_factor = warmup_start_factor
        self.adam_beta1 = adam_beta1
        self.adam_beta2 = adam_beta2
        self.cosine_t0 = cosine_t0
        self.cosine_tmult = cosine_tmult

        self.criterion = criterion
        self.bce = torch.nn.BCEWithLogitsLoss()
        self.ap = CustomMultiLabelAveragePrecision(num_labels=self.num_classes)
        self.f1 = MultilabelF1Score(num_labels=self.num_classes, average="macro")
        self.f2 = MultilabelFBetaScore(
            num_labels=self.num_classes, beta=2.0, average="macro"
        )

        if callable(getattr(self.criterion, "set_device", None)):
            self.criterion.set_device(self.device)

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
        self.log(
            "val_ap", self.ap.compute(), on_step=False, on_epoch=True, prog_bar=True
        )
        self.ap.reset()
        self.log(
            "val_f1", self.f1.compute(), on_step=False, on_epoch=True, prog_bar=True
        )
        self.f1.reset()
        self.log(
            "val_f2", self.f2.compute(), on_step=False, on_epoch=True, prog_bar=True
        )
        self.f2.reset()

    def test_step(self, batch, batch_idx):
        x, y, _ = batch
        logits = self(x)
        loss = self.criterion(logits, y.float())
        self.log("test_loss", loss)
        return loss

    def configure_optimizers(self):
        if self.optimizer_type == "sgd":
            optim = torch.optim.SGD(
                self.parameters(),
                lr=self.learning_rate,
                momentum=self.momentum,
                weight_decay=self.weight_decay,
            )
            # Use step decay schedule for SGD
            base_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optim,
                milestones=self.lr_steps,
                gamma=self.lr_decay,
            )
        elif self.optimizer_type == "adamw":
            optim = torch.optim.AdamW(
                self.parameters(),
                lr=self.learning_rate,
                betas=(self.adam_beta1, self.adam_beta2),
                weight_decay=self.weight_decay,
            )
            # Use cosine decay schedule for AdamW
            base_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optim, T_0=self.cosine_t0, T_mult=self.cosine_tmult, eta_min=self.min_lr
            )
        else:
            raise ValueError(f"Unsupported optimizer type: {self.optimizer_type}")

        # Add warmup to either scheduler type
        if self.warmup_steps > 0:
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                optim,
                start_factor=self.warmup_start_factor,
                end_factor=1.0,
                total_iters=self.warmup_steps,
            )
            seq_scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer=optim,
                schedulers=[warmup_scheduler, base_scheduler],
                milestones=[self.warmup_steps],
            )
            scheduler = {
                "scheduler": seq_scheduler,
                "interval": "epoch",
                "frequency": 1,
            }
        else:
            # No warmup
            scheduler = {
                "scheduler": base_scheduler,
                "interval": "epoch",
                "frequency": 1,
            }

        return {"optimizer": optim, "lr_scheduler": scheduler}
