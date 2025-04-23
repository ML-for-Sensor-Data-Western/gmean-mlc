import os
from typing import Literal

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader

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
        annRoot: str,
        imgRoot: str,
        split: Literal["Train", "Val", "Test"] = "Train",
        transform=None,
        loader=default_loader,
        onlyDefects=False,
    ):
        if split not in ["Train", "Val", "Test"]:
            raise ValueError(
                f"Split must be one of 'Train', 'Val', or 'Test', got {split}"
            )
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

        return img, None, path # None for targets


if __name__ == "__main__":
    import torchvision.transforms as transforms

    transform = transforms.Compose(
        [transforms.Resize((224, 224)), transforms.ToTensor()]
    )

    train = MultiLabelDataset(
        annRoot="/mnt/datassd0/sewer-data/annotations",
        imgRoot="/mnt/datassd0/sewer-data/images",
        split="Train",
        transform=transform,
    )

    val = MultiLabelDataset(
        annRoot="/mnt/datassd0/sewer-data/annotations",
        imgRoot="/mnt/datassd0/sewer-data/images",
        split="Val",
        transform=transform,
    )

    print("Number of classes: ", train.num_classes)

    print(
        f"\nTraining Set:"
        f"\nNumber of samples: {len(train)}"
        f"\nClass counts: {train.class_counts}"
        f"\nAny class count: {train.any_class_count}"
        f"\nNegative count: {len(train) - train.any_class_count}"
    )

    print(
        f"\nValidation Set:"
        f"\nNumber of samples: {len(val)}"
        f"\nClass counts: {val.class_counts}"
        f"\nAny class count: {val.any_class_count}"
        f"\nNegative count: {len(val) - val.any_class_count}"
    )

    # plot class count percentages in a bar plot, cover each split in different color
    import matplotlib.pyplot as plt

    # Get class counts for each split
    train_counts = train.class_counts.numpy()
    val_counts = val.class_counts.numpy()

    # add negative count to each split
    train_counts = np.append(train_counts, len(train) - train.any_class_count)
    val_counts = np.append(val_counts, len(val) - val.any_class_count)

    # Calculate percentages
    train_percentages = train_counts / len(train)
    val_percentages = val_counts / len(val)

    # Plot percentages for each split
    plt.figure(figsize=(14, 6))
    bar_width = 0.25
    class_names = train.LabelNames + ["Negative"]
    indices = np.arange(len(class_names))

    plt.bar(
        indices,
        train_percentages,
        width=bar_width,
        label="Training",
        alpha=0.7,
        color="blue",
    )

    plt.bar(
        indices + bar_width,
        val_percentages,
        width=bar_width,
        label="Validation",
        alpha=0.7,
        color="green",
    )

    # Add labels and title
    plt.xlabel("Class Index")
    plt.ylabel("Percentage of Samples")
    plt.title("Class Count Percentages for Training and Validation Sets")
    plt.xticks(indices + bar_width, class_names, rotation=90)
    plt.legend()

    plt.savefig("class_count_percentages_sewer.png")
