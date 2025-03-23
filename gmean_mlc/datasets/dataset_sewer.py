import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader

__all__ = ["MultiLabelDataset", "MultiLabelDatasetInference"]

Labels = [
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
    "VA",
    "ND",
]


class MultiLabelDataset(Dataset):
    def __init__(
        self,
        annRoot,
        imgRoot,
        split="Train",
        transform=None,
        loader=default_loader,
        onlyDefects=False,
    ):
        super(MultiLabelDataset, self).__init__()
        self.imgRoot = imgRoot
        self.annRoot = annRoot
        self.split = split

        self.transform = transform
        self.loader = default_loader

        self.LabelNames = Labels.copy()
        self.LabelNames.remove("VA")
        self.LabelNames.remove("ND")
        self.onlyDefects = onlyDefects

        self.num_classes = len(self.LabelNames)
        self.img_paths, self.labels, self.class_counts, self.any_class_count = (
            self._load_annotations()
        )
        self.num_samples = len(self.img_paths)

    def _load_annotations(self):
        gtPath = os.path.join(self.annRoot, "SewerML_{}.csv".format(self.split))
        gt = pd.read_csv(
            gtPath,
            sep=",",
            encoding="utf-8",
            usecols=self.LabelNames + ["Filename", "Defect"],
        )

        if self.onlyDefects:
            gt = gt[gt["Defect"] == 1]

        img_paths = gt["Filename"].values
        labels = gt[self.LabelNames].values

        class_counts = self._get_class_counts(labels)
        defect_count = self._get_defect_count(gt)

        return img_paths, labels, class_counts, defect_count

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        path = self.img_paths[index]

        img = self.loader(os.path.join(self.imgRoot, path))
        if self.transform is not None:
            img = self.transform(img)

        target = self.labels[index, :]

        return img, target, path

    def _get_class_counts(self, labels: pd.DataFrame):
        """Count the number of samples for each class"""
        class_counts = [
            len(labels[labels[:, defect_idx] == 1])
            for defect_idx in range(self.num_classes)
        ]
        return torch.as_tensor(np.array(class_counts))

    def _get_defect_count(self, gt: pd.DataFrame):
        """Count the number of samples with defects"""
        # new column of 1 if any column has 1, 0 otherwise
        defect_count = gt[gt["Defect"] == 1].shape[0]
        return defect_count


class MultiLabelDatasetInference(Dataset):
    def __init__(
        self,
        annRoot,
        imgRoot,
        split="Train",
        transform=None,
        loader=default_loader,
        onlyDefects=False,
    ):
        super(MultiLabelDatasetInference, self).__init__()
        self.imgRoot = imgRoot
        self.annRoot = annRoot
        self.split = split

        self.transform = transform
        self.loader = default_loader

        self.LabelNames = Labels.copy()
        self.LabelNames.remove("VA")
        self.LabelNames.remove("ND")
        self.onlyDefects = onlyDefects

        self.num_classes = len(self.LabelNames)

        self.loadAnnotations()

    def loadAnnotations(self):
        gtPath = os.path.join(self.annRoot, "SewerML_{}.csv".format(self.split))
        gt = pd.read_csv(gtPath, sep=",", encoding="utf-8", usecols=["Filename"])

        self.img_paths = gt["Filename"].values

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        path = self.img_paths[index]

        img = self.loader(os.path.join(self.imgRoot, path))
        if self.transform is not None:
            img = self.transform(img)

        return img, path


if __name__ == "__main__":
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader

    transform = transforms.Compose(
        [transforms.Resize((224, 224)), transforms.ToTensor()]
    )

    train = MultiLabelDataset(
        annRoot="./annotations", imgRoot="./Data", split="Train", transform=transform
    )
    train_defect = MultiLabelDataset(
        annRoot="./annotations",
        imgRoot="./Data",
        split="Train",
        transform=transform,
        onlyDefects=True,
    )

    print(len(train), len(train_defect))
    print(
        train.class_counts,
        train_defect.class_counts,
    )
