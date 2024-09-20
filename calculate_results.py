import argparse
import json
import os

import pandas as pd

from metrics import evaluation

LABEL_WEIGHT_DICT = {
    "RB": 1.00,
    "OB": 0.5518,
    "PF": 0.2896,
    "DE": 0.1622,
    "FS": 0.6419,
    "IS": 0.1847,
    "RO": 0.3559,
    "IN": 0.3131,
    "AF": 0.0811,
    "BE": 0.2275,
    "FO": 0.2477,
    "GR": 0.0901,
    "PH": 0.4167,
    "PB": 0.4167,
    "OS": 0.9009,
    "OP": 0.3829,
    "OK": 0.4396,
}
LABELS = list(LABEL_WEIGHT_DICT.keys())
LABEL_WEIGHTS = list(LABEL_WEIGHT_DICT.values())


def calcualte_results(scores, targets, score_file, args):
    main_metrics, meta_metrics, class_metrics = evaluation(
        scores, targets, LABEL_WEIGHTS, threshold=args.threshold
    )

    outputName = "{}_{}_{}".format(args.split, score_file[:-4], args.threshold)

    with open(
        os.path.join(args.score_path, "{}.json".format(outputName)),
        "w",
        encoding="utf-8",
    ) as fp:
        json.dump(
            {
                "Highlights": {
                    "CIW_F2": main_metrics["CIW_F2"],
                    "NORMAL_F1": meta_metrics["NORMAL_F1"],
                    "MACRO_F2": main_metrics["MACRO_F2"],
                    "MACRO_F1": main_metrics["MACRO_F1"],
                    "DEFECT_F1": meta_metrics["DEFECT_F1"],
                },
                "Main": main_metrics,
                "Meta": meta_metrics,
                "Class": class_metrics,
                "Labels": LABELS,
                "LabelWeights": LABEL_WEIGHTS,
            },
            fp,
            indent=4,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--split", type=str, default="Val", choices=["Train", "Val", "Test"]
    )
    parser.add_argument("--score_path", type=str, default="./results")
    parser.add_argument("--gt_path", type=str, default="./annotations")

    args = parser.parse_args()

    score_path = args.score_path
    target_path = args.gt_path

    split = args.split

    target_split_path = os.path.join(target_path, "SewerML_{}.csv".format(split))
    targets_df = pd.read_csv(target_split_path, sep=",")
    targets_df = targets_df.sort_values(by=["Filename"]).reset_index(drop=True)
    targets = targets_df[LABELS].values

    for subdir, dirs, files in os.walk(score_path):
        print("Iterating in dir: ", subdir)
        for score_file in files:
            if split.lower() not in score_file:
                continue
            if (
                "e2e" not in score_file
                and "twostage" not in score_file
                and "defect" not in score_file
            ):
                continue
            if not "sigmoid" in score_file:
                continue
            if os.path.splitext(score_file)[-1] != ".csv":
                continue
            print("Calculating results for: ", score_file)

            scores_df = pd.read_csv(os.path.join(subdir, score_file), sep=",")
            scores_df = scores_df.sort_values(by=["Filename"]).reset_index(drop=True)

            scores = scores_df[LABELS].values
            calcualte_results(scores, targets, score_file, args)
