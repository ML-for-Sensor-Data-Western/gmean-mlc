import os
import zipfile
from io import BytesIO
from PIL import Image

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

__all__ = ["MultiLabelDatasetChest", "MultiLabelDatasetInferenceChest"]

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

# Global cache for mapping PNG file names to their zip archive and internal path.
_zip_mapping = None

def build_zip_mapping(zip_folder):
    mapping = {}
    for zip_filename in os.listdir(zip_folder):
        if zip_filename.endswith(".zip"):
            zip_path = os.path.join(zip_folder, zip_filename)
            with zipfile.ZipFile(zip_path, 'r') as z:
                for internal_name in z.namelist():
                    # Get the base file name; if your ZIP archives have folders inside, this will strip them.
                    base = os.path.basename(internal_name)
                    # Avoid overwriting if duplicates occur (or handle as needed)
                    mapping[base] = (zip_path, internal_name)
    return mapping

def custom_zip_loader(image_filename, zip_folder):
    global _zip_mapping
    if _zip_mapping is None:
        _zip_mapping = build_zip_mapping(zip_folder)
    if image_filename not in _zip_mapping:
        raise FileNotFoundError(f"{image_filename} not found in any zip archive in {zip_folder}")
    zip_path, internal_name = _zip_mapping[image_filename]
    with zipfile.ZipFile(zip_path, 'r') as z:
        with z.open(internal_name) as f:
            img = Image.open(BytesIO(f.read()))
            return img.convert('RGB')

class MultiLabelDatasetChest(Dataset):
    def __init__(
        self,
        annRoot,
        imgRoot,
        split="Train",
        transform=None,
        loader=None,
        onlyDefects=False,
    ):
        super(MultiLabelDatasetChest, self).__init__()
        self.imgRoot = imgRoot  # This folder contains the ZIP archives.
        self.annRoot = annRoot
        self.split = split
        self.transform = transform

        # Use custom ZIP loader if no loader is provided.
        if loader is None:
            self.loader = lambda fname: custom_zip_loader(fname, zip_folder=imgRoot)
        else:
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
        # Loader uses the PNG file name from CSV and looks it up in the ZIP mapping.
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
        split="Train",
        transform=None,
        loader=None,
    ):
        super(MultiLabelDatasetInferenceChest, self).__init__()
        self.annRoot = annRoot
        self.imgRoot = imgRoot
        self.split = split
        self.transform = transform
        if loader is None:
            self.loader = lambda fname: custom_zip_loader(fname, zip_folder=imgRoot)
        else:
            self.loader = loader

        self.loadAnnotations()

    def loadAnnotations(self):
        if self.split == "Train":
            list_path = os.path.join(self.annRoot, "train_val_list.txt")
        else:
            list_path = os.path.join(self.annRoot, "test_list.txt")

        with open(list_path, "r") as f:
            self.img_paths = [line.strip() for line in f]

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
    from torch.utils.data import DataLoader

    transform = transforms.Compose(
        [transforms.Resize((224, 224)), transforms.ToTensor()]
    )

    # Create a training dataset.
    # annRoot: folder containing CSV and split text files.
    # imgRoot: folder containing the ZIP archives.
    dataset_train = MultiLabelDatasetChest(
        annRoot="/mnt/datassd0/chest-xray/data/",
        imgRoot="/mnt/datassd0/chest-xray/data/images",
        split="Train",
        transform=transform,
    )

    print("\nTraining Set:")
    print("Number of samples:", len(dataset_train))
    print("Per-class counts:", dataset_train.class_counts.numpy())
    print("Number of samples with any finding:", dataset_train.any_class_count)
    
    
    dataset_test = MultiLabelDatasetChest(
        annRoot="/mnt/datassd0/chest-xray/data/",
        imgRoot="/mnt/datassd0/chest-xray/data/images",
        split="Test",
        transform=transform,
    )
    
    print("\nTest Set:")
    print("Number of samples:", len(dataset_test))
    print("Per-class counts:", dataset_test.class_counts.numpy())
    print("Number of samples with any finding:", dataset_test.any_class_count)
    
    
    # add class wise count train and test
    class_counts = [count_1.numpy() + count_2.numpy() for count_1, count_2 in zip(dataset_train.class_counts, dataset_test.class_counts)]
    print("\nTotal Set:")
    print("Number of samples:", len(dataset_train) + len(dataset_test))
    print("Per-class counts:", class_counts)
    print("Number of samples with any finding:", dataset_train.any_class_count + dataset_test.any_class_count)