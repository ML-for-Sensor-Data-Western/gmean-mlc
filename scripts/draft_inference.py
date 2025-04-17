import os
from argparse import ArgumentParser

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models as torch_models
from torchvision import transforms

from gmean_mlc import models as ml_models
from gmean_mlc.datasets import MultiLabelDataset
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


def load_model(model_path, device):
    """Load the model from the given path
    
    Args:
        model_path (str): The path to the model checkpoint
        device (torch.device): The device to load the model on
        
    Returns:
        lt_model (MultiLabelModel): The loaded model
        model_name (str): The name of the model
    """
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

    lt_model = MultiLabelModel(
        model=model_name,
        num_classes=num_classes,
    )

    lt_model.load_state_dict(model_state_dict)

    lt_model = lt_model.to(device)

    return lt_model, model_name


def evaluate(dataloader, model, device):
    """Evaluate the model on the given dataloader
    
    Args:
        dataloader (DataLoader): The dataloader to evaluate the model on
        model (MultiLabelModel): The model to evaluate
        device (torch.device): The device to evaluate the model on
        
    Returns:
        
    """
    model.eval()
    sigmoid = nn.Sigmoid()
    with torch.no_grad():
        for i, (images, targets, imgPaths) in enumerate(dataloader):
            images = images.to(device)
            output = model(images)
            sigmoidOutput = sigmoid(output).detach().cpu().numpy()
            
            # Batch output. Do something with the predictions


def run_cli():
    parser = ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, default="sewer", choices=["sewer", "coco", "chest"]
    )
    parser.add_argument("--ann_root", type=str, default="./annotations")
    parser.add_argument("--data_root", type=str, default="./Data")
    parser.add_argument("--split", type=str, default="Train", choices=["Train", "Val", "Test"])
    parser.add_argument("--model_path", type=str, default=".example.ckpt")
    parser.add_argument(
        "--device_id", type=int, default=0, help="GPU device ID to use for inference"
    )
    parser.add_argument("--batch_size", type=int, default=128)
    args = vars(parser.parse_args())
    
    ann_root = args["ann_root"]
    data_root = args["data_root"]
    split = args["split"]

    # initialize dataloaders
    img_size = 224

    # transformation
    eval_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=SEWER_MEAN, std=SEWER_STD),
    ])

    # initialize device
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args['device_id']}")
        print(f"Using GPU {args['device_id']}")
    else:
        device = torch.device("cpu")
        print("CUDA not available, using CPU")

    # Run inference for each model maximizing/minimizing the metric
    model_path = args["model_path"]
    lt_model, model_name = load_model(model_path, device)
    
    dataset = MultiLabelDataset(
        ann_root,
        data_root,
        split=split,
        transform=eval_transform,
        onlyDefects=False,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args["batch_size"],
        num_workers=2,
        pin_memory=True,
    )

    labelNames = dataset.LabelNames

    # Validation results
    print(f"Processing {split} split")
    
    # Modify the function to explain the model
    evaluate(dataloader, lt_model, device)


if __name__ == "__main__":
    run_cli()
