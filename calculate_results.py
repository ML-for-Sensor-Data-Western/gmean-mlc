import os
import json
import argparse
import pandas as pd
from metrics import evaluation


LabelWeightDict = {"RB":1.00,"OB":0.5518,"PF":0.2896,"DE":0.1622,"FS":0.6419,"IS":0.1847,"RO":0.3559,"IN":0.3131,"AF":0.0811,"BE":0.2275,"FO":0.2477,"GR":0.0901,"PH":0.4167,"PB":0.4167,"OS":0.9009,"OP":0.3829,"OK":0.4396}
Labels = list(LabelWeightDict.keys())
LabelWeights = list(LabelWeightDict.values())

def calcualteResults(args):
    scorePath = args["score_path"]
    targetPath = args["gt_path"]

    outputPath = args["output_path"]

    if not os.path.isdir(outputPath):
        os.makedirs(outputPath)

    split = args["split"]

    targetSplitpath = os.path.join(targetPath, "SewerML_{}.csv".format(split))
    targetsDf = pd.read_csv(targetSplitpath, sep=",")
    targetsDf = targetsDf.sort_values(by=["Filename"]).reset_index(drop=True)
    targets = targetsDf[Labels].values

    for subdir, dirs, files in os.walk(scorePath):
        print("Iterating in dir: ", subdir)
        for scoreFile in files:
            if split.lower() not in scoreFile:
                continue
            if "e2e" not in scoreFile and "twostage" not in scoreFile and "defect" not in scoreFile:
                continue
            if not "sigmoid" in scoreFile:
                continue
            if os.path.splitext(scoreFile)[-1] != ".csv":
                continue
            print("Calculating results for: ", scoreFile)

            scoresDf = pd.read_csv(os.path.join(subdir, scoreFile), sep=",")
            scoresDf = scoresDf.sort_values(by=["Filename"]).reset_index(drop=True)

            scores = scoresDf[Labels].values

            main_metrics, meta_metrics, class_metrics = evaluation(scores, targets, LabelWeights, threshold=0.5)

            outputName = "{}_{}".format(split, scoreFile)
            if split.lower() == "test":
                outputName = outputName[:len(outputName) - len("_test_sigmoid.csv")]
            elif split.lower() == "val":
                outputName = outputName[:len(outputName) - len("_val_sigmoid.csv")]
            elif split.lower() == "train":
                outputName = outputName[:len(outputName) - len("_train_sigmoid.csv")]


            with open(os.path.join(outputPath,'{}.json'.format(outputName)), 'w') as fp:
                json.dump({"Main": main_metrics, "Meta": meta_metrics, "Class": class_metrics, "Labels": Labels, "LabelWeights": LabelWeights,}, fp, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str, default = "./results")
    parser.add_argument("--split", type=str, default = "Val", choices=["Train", "Val", "Test"])
    parser.add_argument("--score_path", type=str, default = "./results")
    parser.add_argument("--gt_path", type=str, default = "./annotations")

    args = vars(parser.parse_args())

    calcualteResults(args)