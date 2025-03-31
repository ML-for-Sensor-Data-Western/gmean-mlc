import os
from argparse import ArgumentParser
import glob
import re

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models as torch_models
from torchvision import transforms

from gmean_mlc import models as ml_models
from gmean_mlc.datasets import (
    MultiLabelDataset,
    MultiLabelDatasetChest,
    MultiLabelDatasetCoco,
)
from gmean_mlc.lightning_model import MultiLabelModel

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

SEWER_MEAN = [0.523, 0.453, 0.345]
SEWER_STD = [0.210, 0.199, 0.154]

COCO_MEAN = [0.470, 0.447, 0.408]
COCO_STD = [0.233, 0.228, 0.231]

CHEST_MEAN = [0.506, 0.506, 0.506]
CHEST_STD = [0.230, 0.230, 0.230]


def find_best_checkpoint(version_dir, metric):
    """Find the best checkpoint for a given metric in a version directory"""
    pattern = os.path.join(version_dir, f"*{metric}=*.ckpt")
    checkpoints = glob.glob(pattern)
    if not checkpoints:
        return None

    # Extract metric value from filename and find best
    try:
        best_ckpt = max(
            checkpoints,
            key=lambda x: float(
                re.search(f"{metric}=([0-9]+(?:\.[0-9]+)?)", x).group(1)
            ),
        )
        return best_ckpt
    except (AttributeError, ValueError) as e:
        print(
            f"Warning: Could not parse metric value for {metric} in {version_dir}. Error: {e}"
        )
        return None


def evaluate(dataloader, model, device):
    model.eval()

    sigmoidPredictions = None
    imgPathsList = []

    sigmoid = nn.Sigmoid()

    dataLen = len(dataloader)

    with torch.no_grad():
        for i, (images, imgPaths) in enumerate(dataloader):
            if i % 100 == 0:
                print("{} / {}".format(i, dataLen))

            images = images.to(device)

            output = model(images)

            sigmoidOutput = sigmoid(output).detach().cpu().numpy()

            if sigmoidPredictions is None:
                sigmoidPredictions = sigmoidOutput
            else:
                sigmoidPredictions = np.vstack((sigmoidPredictions, sigmoidOutput))

            imgPathsList.extend(list(imgPaths))
    return sigmoidPredictions, imgPathsList


def load_model(model_path, device):
    if not os.path.isfile(model_path):
        raise ValueError(
            "The provided path does not lead to a valid file: {}".format(model_path)
        )

    model_ckpt = torch.load(model_path)

    print("Model Hyperparams: \n", model_ckpt["hyper_parameters"])

    model_name = model_ckpt["hyper_parameters"]["model"]
    if model_name not in MODEL_NAMES:
        raise ValueError(
            "Got model {}, but no such model is in this codebase".format(model_name)
        )
    num_classes = model_ckpt["hyper_parameters"]["num_classes"]

    model_state_dict = model_ckpt["state_dict"]

    keys_to_drop = [
        "biases",
        "criterion.bce_with_weights.pos_weight",
        "criterion.bce_defect_types.pos_weight",
        "criterion.bce_defect.pos_weight",
        "criterion.bce.pos_weight",
    ]
    for key in keys_to_drop:
        if key in model_state_dict.keys():
            model_state_dict.pop(key)

    lt_model = MultiLabelModel(
        model=model_name,
        num_classes=num_classes,
    )

    lt_model.load_state_dict(model_state_dict)

    lt_model = lt_model.to(device)

    return lt_model


def run_inference(args):
    ann_root = args["ann_root"]
    data_root = args["data_root"]
    log_dir = args["log_dir"]
    outputPath = args["results_output"]
    splits = ["Val", "Test"] if args["do_val_test"] else [args["split"]]
    versions = args["versions"]

    if not os.path.isdir(outputPath):
        os.makedirs(outputPath)

    # initialize dataloaders
    img_size = 224

    # normalization parameters
    if args["dataset"] == "sewer":
        data_mean = SEWER_MEAN
        data_std = SEWER_STD
    elif args["dataset"] == "coco":
        data_mean = COCO_MEAN
        data_std = COCO_STD
    elif args["dataset"] == "chest":
        data_mean = CHEST_MEAN
        data_std = CHEST_STD
    else:
        raise Exception("Invalid dataset '{}'".format(args["dataset"]))

    # transformation
    eval_transform_list = [
        transforms.Resize((img_size, img_size)),
    ]
    if args["dataset"] == "chest":
        eval_transform_list.append(transforms.Grayscale(num_output_channels=3))
    eval_transform_list += [
        transforms.ToTensor(),
        transforms.Normalize(mean=data_mean, std=data_std),
    ]
    eval_transform = transforms.Compose(eval_transform_list)

    # dataset class
    if args["dataset"] == "sewer":
        dataset_infer_class = MultiLabelDataset
    elif args["dataset"] == "coco":
        dataset_infer_class = MultiLabelDatasetCoco
    elif args["dataset"] == "chest":
        dataset_infer_class = MultiLabelDatasetChest
    else:
        raise ValueError(f"Invalid dataset '{args['dataset']}'")

    # initialize device
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args['device_id']}")
        print(f"Using GPU {args['device_id']}")
    else:
        device = torch.device("cpu")
        print("CUDA not available, using CPU")

    # Process each version
    for version in versions:
        version_dir = os.path.join(log_dir, f"version_{version}")
        if not os.path.exists(version_dir):
            print(
                f"Warning: Version directory {version_dir} does not exist, skipping..."
            )
            continue

        # Create version-specific output directory
        version_output_dir = os.path.join(outputPath, f"version_{version}")
        if not os.path.isdir(version_output_dir):
            os.makedirs(version_output_dir)

        # Run inference for each model maximizing the metric
        for metric in ["val_f1", "val_f2", "val_ap"]:
            model_path = find_best_checkpoint(version_dir, metric)
            if not model_path:
                print(
                    f"Warning: No checkpoint found for metric {metric} in {version_dir}, skipping..."
                )
                continue

            print(f"\nProcessing version {version} with {metric} checkpoint")
            lt_model = load_model(model_path, device)

            # if multiple splits (Val, Test)
            for split in splits:
                dataset = dataset_infer_class(
                    ann_root,
                    data_root,
                    split=split,
                    transform=eval_transform,
                    onlyDefects=False,
                )
                dataloader = DataLoader(
                    dataset,
                    batch_size=args["batch_size"],
                    num_workers=args["workers"],
                    pin_memory=True,
                )

                labelNames = dataset.LabelNames

                # Validation results
                print(f"Processing {split} split")
                sigmoid_predictions, val_imgPaths = evaluate(
                    dataloader, lt_model, device
                )

                sigmoid_dict = {}
                sigmoid_dict["Filename"] = val_imgPaths
                for idx, header in enumerate(labelNames):
                    sigmoid_dict[header] = sigmoid_predictions[:, idx]

                sigmoid_df = pd.DataFrame(sigmoid_dict)
                # Include metric in filename
                metric_name = metric.replace("val_", "")
                sigmoid_df.to_csv(
                    os.path.join(
                        version_output_dir,
                        f"{lt_model.model}_{metric_name}_{split.lower()}_sigmoid.csv",
                    ),
                    sep=",",
                    index=False,
                )


def run_cli():
    parser = ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, default="sewer", choices=["sewer", "coco", "chest"]
    )
    parser.add_argument("--ann_root", type=str, default="./annotations")
    parser.add_argument("--data_root", type=str, default="./Data")
    parser.add_argument(
        "--batch_size", type=int, default=512, help="Size of the batch per GPU"
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--log_dir",
        type=str,
        required=True,
        help="Directory containing version folders (e.g., ./logs)",
    )
    parser.add_argument(
        "--versions",
        nargs="+",
        type=str,
        required=True,
        help="List of version numbers/names to process (e.g., 1 2 10_special)",
    )
    parser.add_argument("--results_output", type=str, default="./results")
    parser.add_argument(
        "--split", type=str, default="Val", choices=["Train", "Val", "Test"]
    )
    parser.add_argument(
        "--do_val_test",
        action="store_true",
        help="If true, inference on both val and test sets.",
    )
    parser.add_argument(
        "--device_id", type=int, default=0, help="GPU device ID to use for inference"
    )

    args = vars(parser.parse_args())
    run_inference(args)


if __name__ == "__main__":
    run_cli()
