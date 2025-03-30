import os

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
        annRoot,
        imgRoot,
        split="Train",
        transform=None,
        loader=default_loader,
        onlyDefects=False,
    ):
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
        csv_path = os.path.join(self.annRoot, "Data_Entry_2017_v2020.csv")
        df = pd.read_csv(csv_path)

        if self.split == "Train":
            list_path = os.path.join(self.annRoot, "train_val_list.txt")
        else:
            list_path = os.path.join(self.annRoot, "test_list.txt")

        with open(list_path, "r") as f:
            keep_filenames = set(line.strip() for line in f)

        # Keep only rows corresponding to the split.
        df = df[df["Image Index"].isin(keep_filenames)].reset_index(drop=True)
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


class MultiLabelDatasetInferenceChest(Dataset):
    def __init__(
        self,
        annRoot,
        imgRoot,
        split="Test",
        transform=None,
        loader=None,
    ):
        super(MultiLabelDatasetInferenceChest, self).__init__()
        self.annRoot = annRoot
        self.imgRoot = imgRoot
        self.split = split
        self.transform = transform
        self.loader = loader
        
        self.LabelNames = Labels.copy()

        self.loadAnnotations()

    def loadAnnotations(self):
        if self.split == "Train":
            list_path = os.path.join(self.annRoot, "train_val_list.txt")
        else:
            list_path = os.path.join(self.annRoot, "test_list.txt")

        with open(list_path, "r") as f:
            img_paths = [line.strip() for line in f]
            self.img_paths = [os.path.join(self.imgRoot, img_path) for img_path in img_paths]

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_filename = self.img_paths[index]
        img = self.loader(img_filename)

        if self.transform is not None:
            img = self.transform(img)

        return img, img_filename


if __name__ == "__main__":
    import torchvision.transforms as transforms

    transform = transforms.Compose(
        [transforms.Resize((224, 224)), transforms.ToTensor()]
    )

    # Create a training dataset.
    # annRoot: folder containing CSV and split text files.
    # imgRoot: folder containing the images.
    train_dataset = MultiLabelDatasetChest(
        annRoot="/mnt/datassd0/chest-xray/data/",
        imgRoot="/mnt/datassd0/chest-xray/data/images/all_images",
        split="Train",
        transform=transform,
    )
    
    test_dataset = MultiLabelDatasetChest(
        annRoot="/mnt/datassd0/chest-xray/data/",
        imgRoot="/mnt/datassd0/chest-xray/data/images/all_images",
        split="Test",
        transform=transform,
    )
    
    print("\nTraining Set:")
    print("Number of samples:", len(train_dataset))
    print("Per-class counts:", train_dataset.class_counts)
    print("Number of samples with any finding:", train_dataset.any_class_count)
    
    print("\nTesting Set:")
    print("Number of samples:", len(test_dataset))
    print("Per-class counts:", test_dataset.class_counts)
    print("Number of samples with any finding:", test_dataset.any_class_count)
    
    # add class wise count train and test
    class_counts = [count_1.numpy() + count_2.numpy() for count_1, count_2 in zip(train_dataset.class_counts, test_dataset.class_counts)]
    print("\nTotal Set:")
    print("Number of samples:", len(train_dataset) + len(test_dataset))
    print("Per-class counts:", class_counts)
    print("Number of samples with any finding:", train_dataset.any_class_count + test_dataset.any_class_count)