"""
This script is used to calculate the results of the sewer dataset.
To use result calculation with any dataser, run calculate_results.py
"""	
import argparse
import json
import os
import warnings

import pandas as pd
from matplotlib import pyplot as plt

from gmean_mlc.metrics.test_metrics import calculate_all_metrics, maximize_class_wise_f_score
from gmean_mlc.utils.constants import SEWER_LABEL_WEIGHT_DICT

SEWER_LABEL_WEIGHTS = list(SEWER_LABEL_WEIGHT_DICT.values())
SEWER_LABELS = list(SEWER_LABEL_WEIGHT_DICT.keys())

def calculate_results_thresholds(scores, targets, output_file):
    num_class = scores.shape[1]

    thresholds = [i / 100 for i in range(5, 100, 5)]
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
        main_metrics_t, meta_metrics_t, class_metrics_t = calculate_all_metrics(
            scores, targets, SEWER_LABEL_WEIGHTS, threshold=threshold
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
    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 2, 3)
    ax3 = fig.add_subplot(2, 2, 4)

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
    plt.savefig(output_file)
    plt.close()


def find_best_val_thresholds_and_calculate_test_results(
    val_scores, test_scores, val_targets, test_targets, output_file, args
):
    """Tune thresholds for each class on validation scores and calculate results on test scores"""
    max_val_f, max_val_t = maximize_class_wise_f_score(
        val_scores, val_targets, args.f_beta
    )

    print("Max F{}: {}".format(args.f_beta, max_val_f))
    print("Max F{} Thresholds: {}".format(args.f_beta, max_val_t))
    
    main_metrics, meta_metrics, class_metrics = calculate_all_metrics(
        test_scores, test_targets, SEWER_LABEL_WEIGHTS, max_val_t
    )
    
    save_results_to_json(
        main_metrics, meta_metrics, class_metrics, output_file, args, max_val_f, list(max_val_t)
    )


def calcualte_results(scores, targets, output_file, args):
    main_metrics, meta_metrics, class_metrics = calculate_all_metrics(
        scores, targets, SEWER_LABEL_WEIGHTS, threshold=args.threshold
    )
    save_results_to_json(main_metrics, meta_metrics, class_metrics, output_file, args)


def load_scores(score_path: str, labels: list[str]):
    """ "Load Scores (or targets) from a csv file"""
    scores_df = pd.read_csv(score_path, sep=",")
    scores_df = scores_df.sort_values(by=["Filename"]).reset_index(drop=True)
    scores = scores_df[labels].values
    return scores


def save_results_to_json(
    main_metrics: dict, 
    meta_metrics: dict, 
    class_metrics: dict, 
    output_file: str, 
    args: argparse.Namespace,
    max_val_f: float = None,
    max_val_t: list = None,
    ):
    result_dict = {
        "Highlights": {
            "MACRO_F1": main_metrics["MACRO_F1"],
            "MACRO_F2": main_metrics["MACRO_F2"],
            "CIW_F2": main_metrics["CIW_F2"],
            "MAP": main_metrics["mAP"],
            "NORMAL_F1": meta_metrics["NORMAL_F1"],
            "DEFECT_F1": meta_metrics["DEFECT_F1"],
        },
        "Main": main_metrics,
        "Meta": meta_metrics,
        "Class": class_metrics,
        "Labels": SEWER_LABELS,
        "LabelWeights": SEWER_LABEL_WEIGHTS,
    }
    
    if args.max_fbeta:
        result_dict["Highlights"][f"VAL_MAX_F{args.f_beta}"] = max_val_f
        result_dict["Highlights"][f"VAL_MAX_F{args.f_beta}_THRESHOLDS"] = max_val_t

    with open(output_file, "w", encoding="utf-8") as fp:
        json.dump(result_dict, fp, indent=4)
         

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_path", type=str, default="./annotations")
    parser.add_argument(
        "--split", type=str, default="Val", choices=["Train", "Val", "Test"]
    )
    parser.add_argument("--score_path", type=str, default="./results")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--multi_threshold", action="store_true")
    parser.add_argument(
        "--max_fbeta",
        action="store_true",
        help="Find best class thresholds to max Fbeta",
    )
    parser.add_argument(
        "--f_beta", type=float, default=1.0, help="beta value to maximize Fbeta"
    )
    parser.add_argument("--val_score_filename", type=str, default="e2e_sigmoid_val.csv")
    parser.add_argument("--test_score_filename", type=str, default="e2e_sigmoid_test.csv")

    args = parser.parse_args()

    if args.max_fbeta:
        warnings.warn(
            "max_fbeta selected. Ignoring threshold, multi-threshold, and split commands."
        )

        val_targets = load_scores(os.path.join(args.gt_path, "SewerML_Val.csv"), SEWER_LABELS)
        test_targets = load_scores(os.path.join(args.gt_path, "SewerML_Test.csv"), SEWER_LABELS)

        val_scores = load_scores(os.path.join(args.score_path, args.val_score_filename), SEWER_LABELS)
        test_scores = load_scores(os.path.join(args.score_path, args.test_score_filename), SEWER_LABELS)

        output_file = os.path.join(
            args.score_path,
            f"Test_{args.test_score_filename[:-4]}_maxbeta_{args.f_beta}.json",
        )
        find_best_val_thresholds_and_calculate_test_results(
            val_scores, test_scores, val_targets, test_targets, output_file, args
        )

    else:
        target_path = os.path.join(args.gt_path, "SewerML_{}.csv".format(args.split))
        targets = load_scores(target_path, SEWER_LABELS)

        for subdir, dirs, files in os.walk(args.score_path):
            print("Iterating in dir: ", subdir)
            for score_file in files:
                if args.split.lower() not in score_file:
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
                score_path = os.path.join(subdir, score_file)
                scores = load_scores(score_path, SEWER_LABELS)

                if args.multi_threshold:
                    output_file = os.path.join(
                        args.score_path, f"{args.split}_{score_file[:-4]}_metrics.png"
                    )
                    calculate_results_thresholds(scores, targets, output_file)
                else:
                    output_file = os.path.join(
                        args.score_path,
                        f"{args.split}_{score_file[:-4]}_{args.threshold}.json",
                    )
                    calcualte_results(scores, targets, output_file, args)
