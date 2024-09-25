import argparse
import json
import os

import pandas as pd
from matplotlib import pyplot as plt

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


def calculate_results_thresholds(scores, targets, score_file, args):
    num_class = scores.shape[1]
    
    thresholds = [i/100 for i in range(5, 100, 5)]
    macro_f1 = []
    macro_f2 = []
    ciw_f2 = []
    normal_f1 = []
    defect_f1 = []
    
    class_f1 = [[] for i in range(num_class)]
    class_f2 = [[] for i in range(num_class)]
    
    for threshold in thresholds:
        print("Calculating results with threshold: ", threshold)
        
        # Assuming evaluation function is defined elsewhere
        main_metrics_t, meta_metrics_t, class_metrics_t = evaluation(
            scores, targets, LABEL_WEIGHTS, threshold=threshold
        )
        
        macro_f1.append(main_metrics_t["MACRO_F1"])
        macro_f2.append(main_metrics_t["MACRO_F2"])
        ciw_f2.append(main_metrics_t["CIW_F2"])
        normal_f1.append(meta_metrics_t["NORMAL_F1"])
        defect_f1.append(meta_metrics_t["DEFECT_F1"])
        
        for class_i in range(num_class):
            class_f1[class_i].append(class_metrics_t["F1_CLS"][class_i])
            class_f2[class_i].append(class_metrics_t["F2_CLS"][class_i])
  
    fig = plt.figure(figsize=(10, 10))
    ax1 = fig.add_subplot(2,1,1)
    ax2 = fig.add_subplot(2,2,3)
    ax3 = fig.add_subplot(2,2,4)
    
    # Plot F1, F2, ciw-F2 in one graph
    ax1.plot(thresholds, macro_f1, label="Macro F1")
    ax1.plot(thresholds, macro_f2, label="Macro F2")
    ax1.plot(thresholds, ciw_f2, label="CIW F2")
    ax1.plot(thresholds, normal_f1, label="Normal F1")
    ax1.plot(thresholds, defect_f1, label="Defect F1")
    
    ax1.set_xlabel("Threshold")
    ax1.set_ylabel("Value")
    ax1.legend()
    ax1.set_title("Overall Metrics vs Threshold")
    ax1.set_ylim(0, 1)
    ax1.grid()
    
    for class_i in range(num_class):
        ax2.plot(thresholds, class_f1[class_i], label=f"F1_{class_i}")
    ax2.set_xlabel("Threshold")
    ax2.set_ylabel("Value")
    ax2.legend()
    ax2.set_title(f"class F1 vs Threshold")
    ax2.set_ylim(0, 1)
    ax2.grid()
    
    for class_i in range(num_class):
        ax3.plot(thresholds, class_f2[class_i], label=f"F2_{class_i}")
    ax3.set_xlabel("Threshold")
    ax3.set_ylabel("Value")
    ax3.legend()
    ax3.set_title(f"class F2 vs Threshold")
    ax3.set_ylim(0, 1)
    ax3.grid()
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.score_path, f"{args.split}_{score_file[:-4]}_metrics.png"))
    plt.close()


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
    parser.add_argument("--gt_path", type=str, default="./annotations")
    parser.add_argument(
        "--split", type=str, default="Val", choices=["Train", "Val", "Test"]
    )
    parser.add_argument("--score_path", type=str, default="./results")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--multi_threshold", action="store_true")  

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
            if args.multi_threshold:
                calculate_results_thresholds(scores, targets, score_file, args)
            else:
                calcualte_results(scores, targets, score_file, args)