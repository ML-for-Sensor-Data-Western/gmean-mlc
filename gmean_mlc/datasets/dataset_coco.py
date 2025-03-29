import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader

_all__ = ["MultiLabelDatasetCoco", "MultiLabelDatasetInferenceCoco"]

class MultiLabelDatasetCoco(Dataset):
    def __init__(
        self,
        annRoot,
        imgRoot,
        split="Train",
        transform=None,
        loader=default_loader,
        onlyDefects=False,
    ):
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
            self.annRoot, "instances_{}2017_balanced.json".format(self.split.lower())
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
        self.labelNames = [self.category_id_to_name[i+1] for i in range(self.num_classes)]

        return

    def _generate_labels(self):
        # Generate binary encoded labels for each image and return as NumPy array.
        labels = np.zeros((len(self.image_ids), self.num_classes))
        for i, img_id in enumerate(self.image_ids):
            for ann in self.annotations[img_id]:
                category_id = ann["category_id"]
                label_index = self.category_map[category_id] - 1
                if label_index < self.num_classes:  # ignore normal class
                    labels[i][label_index] = 1  # Convert to binary
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


class MultiLabelDatasetInferenceCoco(Dataset):
    def __init__(
        self,
        annRoot,
        imgRoot,
        split="Train",
        transform=None,
        loader=default_loader,
        onlyDefects=False,
    ):
        super(MultiLabelDatasetInferenceCoco, self).__init__()
        self.imgRoot = imgRoot
        self.annRoot = annRoot
        self.split = split

        self.transform = transform
        self.loader = default_loader

        self.onlyDefects = onlyDefects

        self._load_annotations()
        self.num_samples = len(self.img_paths)

    def _load_annotations(self):
        gtPath = os.path.join(
            self.annRoot, "instances_{}2017_balanced.json".format(self.split.lower())
        )
        with open(gtPath, "r") as f:
            coco_data = json.load(f)

        self.img_paths = {img["id"]: img["file_name"] for img in coco_data["images"]}
        self.image_ids = list(self.img_paths.keys())

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_id = self.image_ids[index]
        path = self.img_paths[img_id]

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

    train = MultiLabelDatasetCoco(
        annRoot="/mnt/datassd0/coco-2017/annotations/",
        imgRoot="/mnt/datassd0/coco-2017/images/all_images",
        split="Train",
        transform=transform,
    )
    
    val = MultiLabelDatasetCoco(
        annRoot="/mnt/datassd0/coco-2017/annotations/",
        imgRoot="/mnt/datassd0/coco-2017/images/all_images",
        split="Val",
        transform=transform,
    )

    print("Number of classes: ", train.num_classes)
    
    print(
        f"\nTraining Set:" 
        f"\nNumber of samples: {len(train)}"
        f"\nClass counts: {train.class_counts}"
        f"\nAny class count: {train.any_class_count}"
    )
    
    print(
        f"\nValidation Set:" 
        f"\nNumber of samples: {len(val)}"
        f"\nClass counts: {val.class_counts}"
        f"\nAny class count: {val.any_class_count}"
    )