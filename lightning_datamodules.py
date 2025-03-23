import lightning.pytorch as pl
from torch.utils.data import DataLoader

from gmean_mlc.datasets import (
    MultiLabelDataset,
    MultiLabelDatasetCoco,
    MultiLabelDatasetChest,
)


class MultiLabelDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset="sewer",
        batch_size=32,
        workers=4,
        ann_root="./annotations",
        data_root="./Data",
        only_defects=False,
        train_transform=None,
        eval_transform=None,
    ):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.workers = workers
        self.ann_root = ann_root
        self.data_root = data_root
        self.only_defects = only_defects

        self.train_transform = train_transform
        self.eval_transform = eval_transform

    # OPTIONAL, called only on 1 GPU/machine
    def prepare_data(self):
        pass

    # OPTIONAL, called for every GPU/machine (assigning state is OK)
    def setup(self, stage):
        # split dataset
        if self.dataset == "sewer":
            dataset_cls = MultiLabelDataset
        elif self.dataset == "coco":
            dataset_cls = MultiLabelDatasetCoco
        elif self.dataset == "chest":
            dataset_cls = MultiLabelDatasetChest
        else:
            raise ValueError(f"Invalid dataset '{self.dataset}'")
        
        if stage == "fit":
            self.train_dataset = dataset_cls(
                self.ann_root,
                self.data_root,
                split="Train",
                transform=self.train_transform,
                onlyDefects=self.only_defects,
            )
            self.val_dataset = dataset_cls(
                self.ann_root,
                self.data_root,
                split="Val",
                transform=self.eval_transform,
                onlyDefects=self.only_defects,
            )
        if stage == "test":
            self.test_dataset = dataset_cls(
                self.ann_root,
                self.data_root,
                split="Test",
                transform=self.eval_transform,
                onlyDefects=self.only_defects,
            )

        self.num_classes = self.train_dataset.num_classes
        self.num_train_samples = self.train_dataset.num_samples
        self.class_counts = self.train_dataset.class_counts
        self.any_class_count = self.train_dataset.any_class_count

    # return the dataloader for each split
    def train_dataloader(self):
        train_dl = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=True,
        )
        return train_dl

    def val_dataloader(self):
        val_dl = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.workers,
            pin_memory=True,
            persistent_workers=True,
        )
        return val_dl

    def test_dataloader(self):
        test_dl = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.workers,
            pin_memory=True,
        )
        return test_dl