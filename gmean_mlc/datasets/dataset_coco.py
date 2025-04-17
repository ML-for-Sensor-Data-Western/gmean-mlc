import json
import os
from typing import Literal

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader


class MultiLabelDatasetCoco(Dataset):
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
        super(MultiLabelDatasetCoco, self).__init__()
        self.imgRoot = imgRoot
        self.annRoot = annRoot
        self.split = split

        self.transform = transform
        self.loader = default_loader

        self.onlyDefects = onlyDefects

        self._load_annotations()
        self.num_samples = len(self.img_paths)
        self.class_counts = self._get_class_counts(self.labels)
        self.any_class_count = self._get_any_class_count(self.labels)

    def _load_annotations(self):
        gtPath = os.path.join(
            self.annRoot, "instances_{}.json".format(self.split.lower())
        )
        with open(gtPath, "r") as f:
            coco_data = json.load(f)

        # Store image paths and IDs
        self.img_paths = {img["id"]: img["file_name"] for img in coco_data["images"]}
        self.image_ids = list(self.img_paths.keys())

        self.num_classes = len(coco_data["categories"]) - 1  # ignore normal class

        # Store category names along with ID mappings
        self.category_map = {
            c["id"]: i for i, c in enumerate(coco_data["categories"], start=1)
        }
        self.category_id_to_name = {c["id"]: c["name"] for c in coco_data["categories"]}

        # Store annotations
        self.annotations = {img_id: [] for img_id in self.img_paths}
        for ann in coco_data["annotations"]:
            self.annotations[ann["image_id"]].append(ann)

        # Generate labels
        self.labels = self._generate_labels()
        self.LabelNames = [
            self.category_id_to_name[i + 1] for i in range(self.num_classes)
        ]

        return

    def _generate_labels(self):
        # Generate binary encoded labels for each image and return as NumPy array.
        labels = np.zeros((len(self.image_ids), self.num_classes))
        for i, img_id in enumerate(self.image_ids):
            for ann in self.annotations[img_id]:
                category_id = ann["category_id"]
                label_index = self.category_map[category_id] - 1
                if label_index < self.num_classes:  # ignore normal class
                    labels[i][label_index] = 1  # Convert zero label to one
        return labels

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_id = self.image_ids[index]
        path = self.img_paths[img_id]

        img = self.loader(os.path.join(self.imgRoot, path))
        if self.transform is not None:
            img = self.transform(img)

        target = self.labels[index, :]

        return img, target, path

    def _get_class_counts(self, labels: np.ndarray):
        """Count the number of samples for each class"""
        class_counts = labels.sum(axis=0)
        return torch.as_tensor(class_counts)

    def _get_any_class_count(self, labels: np.ndarray):
        """Count the number of samples that have at least one class"""
        any_class_count = len(labels[labels.sum(axis=1) > 0])
        return any_class_count


if __name__ == "__main__":
    import torchvision.transforms as transforms

    transform = transforms.Compose(
        [transforms.Resize((224, 224)), transforms.ToTensor()]
    )

    train = MultiLabelDatasetCoco(
        annRoot="/mnt/datassd0/coco-2017/output_balanced_70_15_15",
        imgRoot="/mnt/datassd0/coco-2017/images/all_images",
        split="Train",
        transform=transform,
    )

    val = MultiLabelDatasetCoco(
        annRoot="/mnt/datassd0/coco-2017/output_balanced_70_15_15",
        imgRoot="/mnt/datassd0/coco-2017/images/all_images",
        split="Val",
        transform=transform,
    )

    test = MultiLabelDatasetCoco(
        annRoot="/mnt/datassd0/coco-2017/output_balanced_70_15_15",
        imgRoot="/mnt/datassd0/coco-2017/images/all_images",
        split="Test",
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

    print(
        f"\nTest Set:"
        f"\nNumber of samples: {len(test)}"
        f"\nClass counts: {test.class_counts}"
        f"\nAny class count: {test.any_class_count}"
        f"\nNegative count: {len(test) - test.any_class_count}"
    )

    # plot class count percentages in a bar plot, cover each split in different color
    import matplotlib.pyplot as plt

    # Get class counts for each split
    train_counts = train.class_counts.numpy()
    val_counts = val.class_counts.numpy()
    test_counts = test.class_counts.numpy()

    # add negative count to each split
    train_counts = np.append(train_counts, len(train) - train.any_class_count)
    val_counts = np.append(val_counts, len(val) - val.any_class_count)
    test_counts = np.append(test_counts, len(test) - test.any_class_count)

    # Calculate percentages
    train_percentages = train_counts / len(train)
    val_percentages = val_counts / len(val)
    test_percentages = test_counts / len(test)

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

    plt.savefig("class_count_percentages_coco.png")
