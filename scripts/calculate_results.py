import argparse
import json
import os
import warnings

import pandas as pd
from matplotlib import pyplot as plt

from gmean_mlc.datasets import (
    MultiLabelDataset,
    MultiLabelDatasetChest,
    MultiLabelDatasetCoco,
)
from gmean_mlc.metrics.test_metrics import (
    calculate_all_metrics,
    maximize_class_wise_f_score,
)
from gmean_mlc.utils.constants import SEWER_LABEL_WEIGHT_DICT


def calculate_results_thresholds(scores, targets, label_weights, output_file):
    num_class = scores.shape[1]

    thresholds = [i / 100 for i in range(5, 100, 5)]
    macro_f1 = []
    macro_f2 = []
    negative_f1 = []
    positive_f1 = []

    class_f1 = [[] for i in range(num_class)]
    class_f2 = [[] for i in range(num_class)]

    for threshold in thresholds:
        print("Calculating results with threshold: ", threshold)

        main_metrics_t, meta_metrics_t, class_metrics_t = calculate_all_metrics(
            scores, targets, label_weights, threshold=threshold
        )

        macro_f1.append(main_metrics_t["MACRO_F1"])
        macro_f2.append(main_metrics_t["MACRO_F2"])
        negative_f1.append(meta_metrics_t["NEGATIVE_F1"])
        positive_f1.append(meta_metrics_t["POSITIVE_F1"])

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
    ax1.plot(thresholds, negative_f1, label="Negative F1")
    ax1.plot(thresholds, positive_f1, label="Positive F1")

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
    # ax2.legend()
    ax2.set_title(f"class F1 vs Threshold")
    ax2.set_ylim(0, 1)
    ax2.grid()

    for class_i in range(num_class):
        ax3.plot(thresholds, class_f2[class_i], label=f"F2_{class_i}")
    ax3.set_xlabel("Threshold")
    ax3.set_ylabel("Value")
    # ax3.legend()
    ax3.set_title(f"class F2 vs Threshold")
    ax3.set_ylim(0, 1)
    ax3.grid()

    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()


def find_best_val_thresholds_and_calculate_test_results(
    val_scores,
    test_scores,
    val_targets,
    test_targets,
    labels,
    label_weights,
    output_file,
    args,
):
    """Tune thresholds for each class on validation scores and calculate results on test scores"""
    max_val_f, max_val_t = maximize_class_wise_f_score(
        val_scores, val_targets, args.f_beta
    )

    print("Max F{}: {}".format(args.f_beta, max_val_f))
    print("Max F{} Thresholds: {}".format(args.f_beta, max_val_t))

    main_metrics, meta_metrics, class_metrics = calculate_all_metrics(
        test_scores, test_targets, label_weights, max_val_t
    )

    save_results_to_json(
        main_metrics,
        meta_metrics,
        class_metrics,
        labels,
        label_weights,
        output_file,
        args,
        max_val_f,
        list(max_val_t),
    )


def calculate_results(scores, targets, labels, label_weights, output_file, args):
    main_metrics, meta_metrics, class_metrics = calculate_all_metrics(
        scores, targets, label_weights, threshold=args.threshold
    )
    save_results_to_json(
        main_metrics,
        meta_metrics,
        class_metrics,
        labels,
        label_weights,
        output_file,
        args,
    )


def load_scores(score_file: str, labels: list[str]):
    """ "Load Scores (or targets) from a csv file"""
    scores_df = pd.read_csv(score_file, sep=",")
    # scores_df = scores_df.sort_values(by=["Filename"]).reset_index(drop=True)
    scores = scores_df[labels].values
    return scores


def save_results_to_json(
    main_metrics: dict,
    meta_metrics: dict,
    class_metrics: dict,
    labels: list[str],
    label_weights: list[float],
    output_file: str,
    args: argparse.Namespace,
    max_val_f: float = None,
    max_val_t: list = None,
):
    result_dict = {
        "Highlights": {
            "MACRO_F1": main_metrics["MACRO_F1"],
            "MACRO_F2": main_metrics["MACRO_F2"],
            "MAP": main_metrics["mAP"],
            "NEGATIVE_F1": meta_metrics["NEGATIVE_F1"],
            "POSITIVE_F1": meta_metrics["POSITIVE_F1"],
            "MACRO_R": main_metrics["MACRO_R"],
        },
        "Main": main_metrics,
        "Meta": meta_metrics,
        "Class": class_metrics,
        "Labels": labels,
    }
    if label_weights is not None:
        result_dict["Highlights"]["CIW_F2"] = main_metrics["CIW_F2"]
        result_dict["LabelWeights"] = label_weights

    if args.max_fbeta:
        result_dict["Highlights"][f"VAL_MAX_F{args.f_beta}"] = max_val_f
        result_dict["Highlights"][f"VAL_MAX_F{args.f_beta}_THRESHOLDS"] = max_val_t

    with open(output_file, "w", encoding="utf-8") as fp:
        json.dump(result_dict, fp, indent=4)


def run_fbeta_maximization(args, dataset_class):
    """
    Run F-beta score maximization using validation set thresholds and evaluate on test set.

    Args:
        args: Argument namespace
        dataset_class: Dataset class to use for loading data
    """
    # Load datasets to get ground truth labels
    val_dataset = dataset_class(
        args.ann_root,
        args.data_root,
        split="Val",
        transform=None,
    )
    test_dataset = dataset_class(
        args.ann_root,
        args.data_root,
        split="Test",
        transform=None,
    )

    # Get label names and ground truth targets
    labels = val_dataset.LabelNames.copy()
    val_targets = val_dataset.labels.copy()
    test_targets = test_dataset.labels.copy()

    label_weights = None
    if args.dataset == "sewer":
        label_weights = list(SEWER_LABEL_WEIGHT_DICT.values())
        # check if Labels from weight dictionary are the same as labels
        if not all(label in SEWER_LABEL_WEIGHT_DICT.keys() for label in labels):
            raise ValueError("Labels from dictionary are not the same as labels")

    # Load model predictions
    val_scores = load_scores(os.path.abspath(args.val_score_filename), labels)
    test_scores = load_scores(os.path.abspath(args.test_score_filename), labels)

    # Define output path and run optimization
    output_file = os.path.join(
        os.path.dirname(os.path.abspath(args.val_score_filename)),
        f"Test_{args.test_score_filename[:-4]}_maxbeta_{args.f_beta}.json",
    )

    find_best_val_thresholds_and_calculate_test_results(
        val_scores,
        test_scores,
        val_targets,
        test_targets,
        labels,
        label_weights,
        output_file,
        args,
    )


def run_multi_path_result_calculation(args, dataset_class):
    """
    Run result calculation on multiple files in the score path directory.

    Args:
        args: Argument namespace
        dataset_class: Dataset class to use for loading data
    """
    # Load dataset to get ground truth labels
    dataset = dataset_class(
        args.ann_root,
        args.data_root,
        split=args.split,
        transform=None,
    )
    labels = dataset.LabelNames.copy()
    targets = dataset.labels.copy()

    label_weights = None
    if args.dataset == "sewer":
        label_weights = list(SEWER_LABEL_WEIGHT_DICT.values())
        # check if Labels from weight dictionary are the same as labels
        if not all(label in SEWER_LABEL_WEIGHT_DICT.keys() for label in labels):
            raise ValueError("Labels from dictionary are not the same as labels")

    for version in args.versions:
        version_dir = os.path.join(args.score_dir, f"version_{version}")
        if not os.path.exists(version_dir):
            print(
                f"Warning: Version directory {version_dir} does not exist, skipping..."
            )
            continue

        # Process each score file in the directory
        for subdir, dirs, files in os.walk(version_dir):
            print("Iterating in dir: ", subdir)
            for score_file in files:
                # Skip files that don't match the split or aren't CSV
                if args.split.lower() not in score_file:
                    continue
                if os.path.splitext(score_file)[-1] != ".csv":
                    continue

                print("Calculating results for: ", score_file)
                score_path = os.path.join(subdir, score_file)
                scores = load_scores(score_path, labels)

                output_file = os.path.join(
                    version_dir,
                    f"{score_file[:-4]}_{args.threshold}.json",
                )
                calculate_results(
                    scores, targets, labels, label_weights, output_file, args
                )

                if args.multi_threshold:
                    output_file = os.path.join(
                        version_dir, f"{score_file[:-4]}_metrics.png"
                    )
                    calculate_results_thresholds(
                        scores, targets, label_weights, output_file
                    )


def main(args):
    # Get appropriate dataset class based on args.dataset
    if args.dataset == "sewer":
        dataset_class = MultiLabelDataset
    elif args.dataset == "coco":
        dataset_class = MultiLabelDatasetCoco
    elif args.dataset == "chest":
        dataset_class = MultiLabelDatasetChest
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    if args.max_fbeta:
        warnings.warn(
            "max_fbeta selected. Ignoring threshold, multi-threshold, and split commands."
        )
        run_fbeta_maximization(args, dataset_class)
    else:
        run_multi_path_result_calculation(args, dataset_class)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, default="sewer", choices=["sewer", "coco", "chest"]
    )
    parser.add_argument("--ann_root", type=str, default="./annotations")
    parser.add_argument("--data_root", type=str, default="./Data")
    parser.add_argument("--split", type=str, default="Val", choices=["Val", "Test"])
    parser.add_argument("--score_dir", type=str, default="./results")
    parser.add_argument(
        "--versions",
        nargs="+",
        type=str,
        required=True,
        help="List of version numbers/names to process (e.g., 1 2 10_special)",
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    # If calculating results for multiple thresholds and plotting them
    parser.add_argument("--multi_threshold", action="store_true")
    # If maximizing Fbeta in Validation set and calculating results on Test set
    parser.add_argument(
        "--max_fbeta",
        action="store_true",
        help="Find best class thresholds to max Fbeta",
    )
    parser.add_argument(
        "--f_beta", type=float, default=1.0, help="beta value to maximize Fbeta"
    )
    parser.add_argument("--val_score_filename", type=str, default="e2e_sigmoid_val.csv")
    parser.add_argument(
        "--test_score_filename", type=str, default="e2e_sigmoid_test.csv"
    )

    args = parser.parse_args()

    main(args)
