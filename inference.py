import os
import numpy as np
from argparse import ArgumentParser
from torchvision import models as torch_models
from torchvision import transforms
from collections import OrderedDict
import pandas as pd
import torch

from dataloader import MultiLabelDatasetInference
from torch.utils.data import DataLoader

import torch.nn as nn

import sewer_models
import ml_models
from lightning_model import MultiLabelModel


TORCHVISION_MODEL_NAMES = sorted(name for name in torch_models.__dict__ if name.islower() and not name.startswith("__") and callable(torch_models.__dict__[name]))
SEWER_MODEL_NAMES = sorted(name for name in sewer_models.__dict__ if name.islower() and not name.startswith("__") and callable(sewer_models.__dict__[name]))
MULTILABEL_MODEL_NAMES = sorted(name for name in ml_models.__dict__ if name.islower() and not name.startswith("__") and callable(ml_models.__dict__[name]))
MODEL_NAMES =  TORCHVISION_MODEL_NAMES + SEWER_MODEL_NAMES + MULTILABEL_MODEL_NAMES


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


def load_model(model_path, best_weights=False):

    if best_weights:
        if not os.path.isfile(model_path):
            raise ValueError("The provided path does not lead to a valid file: {}".format(model_path))
        last_ckpt_path = model_path
    else:
        last_ckpt_path = os.path.join(model_path, "last.ckpt")
        if not os.path.isfile(last_ckpt_path):
            raise ValueError("The provided directory path does not contain a 'last.ckpt' file: {}".format(model_path))
    
    model_last_ckpt = torch.load(last_ckpt_path)
    

    model_name = model_last_ckpt["hyper_parameters"]["model"]
    num_classes = model_last_ckpt["hyper_parameters"]["num_classes"]
    training_mode = model_last_ckpt["hyper_parameters"]["training_mode"]
    br_defect = model_last_ckpt["hyper_parameters"]["br_defect"]
    
    # Load best checkpoint
    best_model = model_last_ckpt
    best_model_state_dict = best_model["state_dict"]
    
    keys_to_drop = ["biases", "criterion.bce_with_weights.pos_weight", "criterion.bce_defect_types.pos_weight", "criterion.bce_defect.pos_weight", "criterion.bce.pos_weight"]
    for key in keys_to_drop:
        if key in best_model_state_dict.keys():
            best_model_state_dict.pop(key)
    # if "biases" in best_model_state_dict.keys():
    #     best_model_state_dict.pop("biases")
    # if "criterion.bce_with_weights.pos_weight" in best_model_state_dict.keys():
    #     best_model_state_dict.pop("criterion.bce_with_weights.pos_weight")

    return best_model_state_dict, model_name, num_classes, training_mode, br_defect


def run_inference(args):

    ann_root = args["ann_root"]
    data_root = args["data_root"]
    model_path = args["model_path"]
    outputPath = args["results_output"]
    best_weights = args["best_weights"]
    splits = ["Val", "Test"] if args["do_val_test"] else [args["split"]]
    
    if not os.path.isdir(outputPath):
        os.makedirs(outputPath)
  
    updated_state_dict, model_name, num_classes, training_mode, br_defect = load_model(model_path, best_weights)
    
    if model_name not in MODEL_NAMES:
        raise ValueError("Got model {}, but no such model is in this codebase".format(model_name))

    if "model_version" not in args.keys():
        model_version = model_name
    else:
        model_version = args["model_version"]

    lt_model = MultiLabelModel(
        model = model_name,
        num_classes=num_classes,
    )

    lt_model.load_state_dict(updated_state_dict)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lt_model = lt_model.to(device)
    
    # initialize dataloaders
    img_size = 299 if model_name in ["inception_v3", "chen2018_multilabel"] else 224
    
    eval_transform=transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.523, 0.453, 0.345], std=[0.210, 0.199, 0.154])
        ])
    
    for split in splits:
        dataset = MultiLabelDatasetInference(ann_root, data_root, split=split, transform=eval_transform, onlyDefects=False)
        dataloader = DataLoader(dataset, batch_size=args["batch_size"], num_workers = args["workers"], pin_memory=True)
        
        labelNames = dataset.LabelNames
        if training_mode == "binary":
            labelNames = ["Defect"]
        elif training_mode == "binaryrelevance":
            labelNames = [br_defect]

        # Validation results
        print("VALIDATION")
        sigmoid_predictions, val_imgPaths = evaluate(dataloader, lt_model, device)

        sigmoid_dict = {}
        sigmoid_dict["Filename"] = val_imgPaths
        for idx, header in enumerate(labelNames):
            sigmoid_dict[header] = sigmoid_predictions[:,idx]

        sigmoid_df = pd.DataFrame(sigmoid_dict)
        sigmoid_df.to_csv(os.path.join(outputPath, "{}_{}_{}_sigmoid.csv".format(model_version, training_mode, split.lower())), sep=",", index=False)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--ann_root', type=str, default='./annotations')
    parser.add_argument('--data_root', type=str, default='./Data')
    parser.add_argument('--batch_size', type=int, default=512, help="Size of the batch per GPU")
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--best_weights", action="store_true", help="If true 'model_path' leads to a specific weight file. If False it leads to the output folder of lightning_trainer where the last.ckpt file is used to read the best model weights.")
    parser.add_argument("--results_output", type=str, default = "./results")
    parser.add_argument("--split", type=str, default = "Val", choices=["Train", "Val", "Test"])
    parser.add_argument("--do_val_test", action="store_true", help="If true, inference on both val and test sets.")

    args = vars(parser.parse_args())

    run_inference(args)