import os
from typing import Literal

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader

# List of labels
Labels = [
    "Infiltration",
    "Effusion",
    "Atelectasis",
    "Nodule",
    "Mass",
    "Pneumothorax",
    "Consolidation",
    "Pleural_Thickening",
    "Cardiomegaly",
    "Emphysema",
    "Edema",
    "Fibrosis",
    "Pneumonia",
    "Hernia",
]


class MultiLabelDatasetChest(Dataset):
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
        super(MultiLabelDatasetChest, self).__init__()
        self.imgRoot = imgRoot
        self.annRoot = annRoot
        self.split = split
        self.transform = transform
        self.loader = loader

        self.LabelNames = Labels.copy()
        self.onlyDefects = onlyDefects

        self.num_classes = len(self.LabelNames)
        self.img_paths, self.labels = self._load_annotations()
        self.num_samples = len(self.img_paths)
        self.class_counts = self._get_class_counts(self.labels)
        self.any_class_count = self._get_any_class_count(self.labels)

    def _load_annotations(self):
        # load csv file with annotations across all splits
        csv_path = os.path.join(self.annRoot, "Data_Entry_2017_v2020.csv")
        df = pd.read_csv(csv_path)

        # Load the filenames for the split
        split_file_list_path = os.path.join(
            self.annRoot, f"train_val_test/{self.split.lower()}.txt"
        )
        with open(split_file_list_path, "r") as f:
            split_filenames = set(line.strip() for line in f)

        # Keep only rows corresponding to the split.
        df = df[df["Image Index"].isin(split_filenames)].reset_index(drop=True)
        # Extract the image file names (should be PNG file names as in the CSV).
        img_paths = df["Image Index"].values
        # add imgRoot to the img_paths
        img_paths = [os.path.join(self.imgRoot, img_path) for img_path in img_paths]

        # Create a multi-hot label matrix.
        labels = np.zeros((len(df), self.num_classes), dtype=int)
        for i, row in df.iterrows():
            findings = row["Finding Labels"].split("|")
            for finding in findings:
                if finding in self.LabelNames:
                    labels[i, self.LabelNames.index(finding)] = 1
                elif finding != "No Finding":
                    raise ValueError(f"Finding {finding} not in LabelNames")
        return img_paths, labels

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        img_filename = self.img_paths[index]
        target = self.labels[index, :]
        img = self.loader(img_filename)

        if self.transform is not None:
            img = self.transform(img)

        return img, torch.FloatTensor(target), img_filename

    def _get_class_counts(self, labels):
        class_counts = [np.sum(labels[:, i] == 1) for i in range(self.num_classes)]
        return torch.as_tensor(np.array(class_counts))

    def _get_any_class_count(self, labels):
        any_count = np.sum(np.any(labels == 1, axis=1))
        return any_count


if __name__ == "__main__":
    import torchvision.transforms as transforms

    transform = transforms.Compose(
        [transforms.Resize((224, 224)), transforms.ToTensor()]
    )

    train_dataset = MultiLabelDatasetChest(
        annRoot="/mnt/datassd0/chest-xray/data/",
        imgRoot="/mnt/datassd0/chest-xray/data/images/all_images",
        split="Train",
        transform=transform,
    )

    val_dataset = MultiLabelDatasetChest(
        annRoot="/mnt/datassd0/chest-xray/data/",
        imgRoot="/mnt/datassd0/chest-xray/data/images/all_images",
        split="Val",
        transform=transform,
    )

    test_dataset = MultiLabelDatasetChest(
        annRoot="/mnt/datassd0/chest-xray/data/",
        imgRoot="/mnt/datassd0/chest-xray/data/images/all_images",
        split="Test",
        transform=transform,
    )

    print("Number of classes: ", train_dataset.num_classes)

    print(
        f"\nTraining Set:"
        f"\nNumber of samples: {len(train_dataset)}"
        f"\nClass counts: {train_dataset.class_counts}"
        f"\nAny class count: {train_dataset.any_class_count}"
        f"\nNegative count: {len(train_dataset) - train_dataset.any_class_count}"
    )

    print(
        f"\nValidation Set:"
        f"\nNumber of samples: {len(val_dataset)}"
        f"\nClass counts: {val_dataset.class_counts}"
        f"\nAny class count: {val_dataset.any_class_count}"
        f"\nNegative count: {len(val_dataset) - val_dataset.any_class_count}"
    )

    print(
        f"\nTesting Set:"
        f"\nNumber of samples: {len(test_dataset)}"
        f"\nClass counts: {test_dataset.class_counts}"
        f"\nAny class count: {test_dataset.any_class_count}"
        f"\nNegative count: {len(test_dataset) - test_dataset.any_class_count}"
    )

    class_counts = (
        train_dataset.class_counts
        + val_dataset.class_counts
        + test_dataset.class_counts
    )
    class_counts = class_counts.numpy()

    print(
        f"\nTotal Set:"
        f"\nNumber of samples: {len(train_dataset) + len(val_dataset) + len(test_dataset)}"
        f"\nClass counts: {class_counts}"
        f"\nAny class count: {train_dataset.any_class_count + val_dataset.any_class_count + test_dataset.any_class_count}"
    )

    # plot class count percentages in a bar plot, cover each split in different color
    import matplotlib.pyplot as plt

    # Get class counts for each split
    train_counts = train_dataset.class_counts.numpy()
    val_counts = val_dataset.class_counts.numpy()
    test_counts = test_dataset.class_counts.numpy()

    # add negative count to each split
    train_counts = np.append(
        train_counts, len(train_dataset) - train_dataset.any_class_count
    )
    val_counts = np.append(val_counts, len(val_dataset) - val_dataset.any_class_count)
    test_counts = np.append(
        test_counts, len(test_dataset) - test_dataset.any_class_count
    )

    # Calculate percentages
    train_percentages = train_counts / len(train_dataset)
    val_percentages = val_counts / len(val_dataset)
    test_percentages = test_counts / len(test_dataset)

    # Plot percentages for each split
    plt.figure(figsize=(14, 6))
    bar_width = 0.25
    class_names = train_dataset.LabelNames + ["Negative"]
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
    plt.bar(
        indices + 2 * bar_width,
        test_percentages,
        width=bar_width,
        label="Testing",
        alpha=0.7,
        color="red",
    )

    # Add labels and title
    plt.xlabel("Class Index")
    plt.ylabel("Percentage of Samples")
    plt.title("Class Count Percentages by Split")
    plt.xticks(indices + bar_width, class_names, rotation=90)
    plt.legend()

    plt.savefig("class_count_percentages_chest.png")
