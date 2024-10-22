import numpy as np
from typing import Tuple, List, Dict

# False Positives = n_p - n_tp
# False Negatives = n_g - n_tp
# True Positives = n_tp
# True Negatives = n_examples - n_p + (n_g - n_tp)


def get_class_counts(
    scores: np.ndarray, targets: np.ndarray, threshold: float | np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate the class counts for binary classification.

    Parameters:
    scores (np.array): The predicted scores for each class.
    targets (np.array): The ground truth labels for each class.
    threshold (float or np.array): Common or class-wise threshold(s)

    Returns:
    n_tp (np.array): The number of true positives for each class.
    n_p (np.array): The total number of positives for each class.
    n_g (np.array): The total number of ground truth occurrences for each class.
    """
    _, n_class = scores.shape

    if isinstance(threshold, float):
        threshold = np.full(n_class, threshold)
    else:
        assert (
            len(threshold) == n_class
        ), "Thresholds must be the same size as the number of classes"

    n_g = np.sum(targets, axis=0)
    n_p = np.sum(scores >= threshold, axis=0)
    n_tp = np.sum((scores >= threshold) * targets, axis=0)

    # If Np is 0 for any class, set to 1 to avoid division with 0
    n_p[n_p == 0] = 1

    return n_tp, n_p, n_g


def get_defect_normal_counts(
    scores: np.ndarray, targets: np.ndarray, threshold: float | np.ndarray
) -> Tuple[int, int, int, int, int, int]:
    """
    Calculates the counts of defect and normal samples based on scores and targets.

    Args:
        scores (np.ndarray): Array of scores.
        targets (np.ndarray): Array of targets.
        threshold (float|np.ndarray): Threshold value(s) for classification.

    Returns:
        Tuple[int, int, int, int, int, int]: A tuple containing the counts of true positives,
        predicted positives, and ground truth positives for defect samples, and the counts of
        true positives, predicted positives, and ground truth positives for normal samples.
    """

    _, n_class = scores.shape

    if isinstance(threshold, float):
        threshold = np.full(n_class, threshold)
    else:
        assert (
            len(threshold) == n_class
        ), "Thresholds must be the same size as the number of classes"

    scores_defect = np.sum(scores >= threshold, axis=1)
    scores_defect[scores_defect > 0] = 1
    scores_normal = np.abs(scores_defect - 1)

    targets_defect = targets.copy()
    # Necessary if using MultiLabelSoftMarginLoss, instead of BCEWithLogitsLoss
    targets_defect[targets == -1] = 0
    targets_defect = np.sum(targets_defect, axis=1)
    targets_defect[targets_defect > 0] = 1
    targets_normal = np.abs(targets_defect - 1)

    n_g_defect = np.sum(targets_defect == 1)
    n_p_defect = np.sum(scores_defect == 1)
    n_tp_defect = np.sum(targets_defect * scores_defect)

    n_g_normal = np.sum(targets_normal == 1)
    n_p_normal = np.sum(scores_normal == 1)
    n_tp_normal = np.sum(targets_normal * scores_normal)

    return n_tp_defect, n_p_defect, n_g_defect, n_tp_normal, n_p_normal, n_g_normal


def get_average_precision(scores, target, max_k=None):
    assert (
        scores.shape == target.shape
    ), "The input and targets do not have the same shape"
    assert (
        scores.ndim == 1
    ), "The input has dimension {}, but expected it to be 1D".format(scores.shape)

    # sort examples
    indices = np.argsort(scores, axis=0)[::-1]

    total_cases = np.sum(target)

    if max_k == None:
        max_k = len(indices)

    # Computes prec@i
    pos_count = 0.0
    total_count = 0.0
    precision_at_i = 0.0

    for i in range(max_k):
        label = target[indices[i]]
        total_count += 1
        if label == 1:
            pos_count += 1
            precision_at_i += pos_count / total_count
        if pos_count == total_cases:
            break

    if pos_count > 0:
        precision_at_i /= pos_count
    else:
        precision_at_i = 0
    return precision_at_i


def get_mean_average_precision(
    scores: np.ndarray, targets: np.ndarray, max_k=None
) -> float:
    """
    Calculate the mean average precision metric.

    Args:
        scores (np.ndarray): Array of predicted scores with shape (n_samples, n_classes).
        targets (np.ndarray): Array of target labels with shape (n_samples, n_classes).
        max_k (int, optional): Maximum number of classes to consider. Defaults to None.

    Returns:
        float: Mean average precision metric.

    """
    _, n_class = scores.shape

    # Array to hold the average precision metric.
    ap = np.zeros(n_class)

    for k in range(n_class):
        scores_k = scores[:, k]
        targets_k = targets[:, k]
        # Necessary if using MultiLabelSoftMarginLoss, instead of BCEWithLogitsLoss
        targets_k[targets_k == -1] = 0

        ap[k] = get_average_precision(scores_k, targets_k)

    return np.mean(ap)



def get_scalar_metrics(n_tp: float, n_p: float, n_g: float):
    p = n_tp / n_p
    r = n_tp / n_g
    f1 = (2 * p * r) / (p + r)
    f2 = (5 * p * r) / (4 * p + r)

    return p, r, f1, f2


def get_micro_metrics(n_g: np.array, n_p: np.array, n_tp: np.array):
    micro_p = np.sum(n_tp) / np.sum(n_p)
    micro_r = np.sum(n_tp) / np.sum(n_g)
    micro_f1 = (2 * micro_p * micro_r) / (micro_p + micro_r)
    micro_f2 = (5 * micro_p * micro_r) / (4 * micro_p + micro_r)

    return micro_p, micro_r, micro_f1, micro_f2


def get_class_metrics(Ng, Np, Nc):
    class_p = Nc / Np
    class_r = Nc / Ng
    class_f1 = (2 * class_p * class_r) / (class_p + class_r)
    class_f2 = (5 * class_p * class_r) / (4 * class_p + class_r)

    class_f1[np.isnan(class_f1)] = 0
    class_f2[np.isnan(class_f2)] = 0

    return class_p, class_r, class_f1, class_f2


def get_macro_metrics(class_p, class_r, class_f1, class_f2):
    n_class = len(class_p)
    macro_p = np.sum(class_p) / n_class
    macro_r = np.sum(class_r) / n_class
    macro_f1 = np.sum(class_f1) / n_class
    macro_f2 = np.sum(class_f2) / n_class

    return macro_p, macro_r, macro_f1, macro_f2


def get_class_weighted_f2(class_f2, weights):
    ciw_f2 = class_f2 * weights
    ciw_f2 = np.sum(ciw_f2) / np.sum(weights)

    return ciw_f2


def evaluation(scores, targets, weights, threshold=0.5):
    assert (
        scores.shape == targets.shape
    ), "The input and targets do not have the same size: Input: {} - Targets: {}".format(
        scores.shape, targets.shape
    )

    n_tp, n_p, n_g = get_class_counts(scores, targets, threshold)

    # Micro Precision, Recall and F1
    micro_p, micro_r, micro_f1, micro_f2 = get_micro_metrics(n_g, n_p, n_tp)

    # Per-Class Precision, Recall and F1
    class_p, class_r, class_f1, class_f2 = get_class_metrics(n_g, n_p, n_tp)

    # Macro metrics
    macro_p, macro_r, macro_f1, macro_f2 = get_macro_metrics(
        class_p, class_r, class_f1, class_f2
    )

    ciw_f2 = get_class_weighted_f2(class_f2, weights)

    # Mean Average Precision (mAP)
    mAP = get_mean_average_precision(scores, targets)

    # Get values for "implict" normal and defect classes
    n_tp_defect, n_p_defect, n_g_defect, n_tp_normal, n_p_normal, n_g_normal = (
        get_defect_normal_counts(scores, targets, threshold)
    )

    # Defect Normal Metrics
    defect_p, defect_r, defect_f1, defect_f2 = get_scalar_metrics(
        n_tp_defect, n_p_defect, n_g_defect
    )
    normal_p, normal_r, normal_f1, normal_f2 = get_scalar_metrics(
        n_tp_normal, n_p_normal, n_g_normal
    )

    # Defect Type classification Metrics
    main_metrics = {
        "MICRO_P": micro_p,
        "MICRO_R": micro_r,
        "MICRO_F1": micro_f1,
        "MICRO_F2": micro_f2,
        "MACRO_P": macro_p,
        "MACRO_R": macro_r,
        "MACRO_F1": macro_f1,
        "MACRO_F2": macro_f2,
        "CIW_F2": ciw_f2,
        "mAP": mAP,
    }

    # Defect/Normal Metrics
    meta_metrics = {
        "DEFECT_P": defect_p,
        "DEFECT_R": defect_r,
        "DEFECT_F1": defect_f1,
        "DEFECT_F2": defect_f2,
        "NORMAL_P": normal_p,
        "NORMAL_R": normal_r,
        "NORMAL_F1": normal_f1,
        "NORMAL_F2": normal_f2,
        "DEFECT_NP": int(n_p_defect),
        "DEFECT_NTP": int(n_tp_defect),
        "DEFECT_NG": int(n_g_defect),
        "NORMAL_NP": int(n_p_normal),
        "NORMAL_NTP": int(n_tp_normal),
        "NORMAL_NG": int(n_g_normal),
    }

    # Class-wise Counts and Metrics
    class_metrics = {
        "P_CLS": list(class_p),
        "R_CLS": list(class_r),
        "F1_CLS": list(class_f1),
        "F2_CLS": list(class_f2),
        "NP": [int(p) for p in n_p],
        "NTP": [int(tp) for tp in n_tp],
        "NG": [int(g) for g in n_g],
    }

    return main_metrics, meta_metrics, class_metrics
