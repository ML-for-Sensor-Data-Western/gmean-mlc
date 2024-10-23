import numpy as np
from typing import Tuple, List, Dict, Union

# False Positives = n_p - n_tp
# False Negatives = n_g - n_tp
# True Positives = n_tp
# True Negatives = n_examples - n_p + (n_g - n_tp)


def precision(
    n_tp: Union[float, np.ndarray], n_p: Union[float, np.ndarray]
) -> Union[float, np.ndarray]:
    """Calculate the precision metric. supports float or np.ndarray"""
    return n_tp / n_p + 1e-10


def recall(
    n_tp: Union[float, np.ndarray], n_g: Union[float, np.ndarray]
) -> Union[float, np.ndarray]:
    """Calculate the recall metric. supports float or np.ndarray"""
    return n_tp / n_g + 1e-10


def fbeta_from_pr(
    p: Union[float, np.ndarray], r: Union[float, np.ndarray], beta: float = 1.0
) -> Union[float, np.ndarray]:
    """Calculate fbeta score from precision and recall. Supports float or np.ndarray"""
    fbeta = (1 + beta**2) * p * r / (beta**2 * p + r)
    return fbeta


def fbeta_score(
    n_g: Union[float, np.ndarray],
    n_p: Union[float, np.ndarray],
    n_tp: Union[float, np.ndarray],
    beta: float = 1.0,
) -> Union[float, np.ndarray]:
    """Calculate fbeta score. supports float or np.ndarray"""
    p = precision(n_tp, n_p)
    r = recall(n_tp, n_g)
    fbeta = fbeta_from_pr(p, r, beta)
    return fbeta


def calculate_class_wise_counts(
    scores: np.ndarray, targets: np.ndarray, threshold: Union[float, np.ndarray]
) -> Tuple[np.ndarray]:
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


def calculate_defect_normal_counts(
    scores: np.ndarray, targets: np.ndarray, threshold: Union[float, np.ndarray]
) -> Tuple[int]:
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


def calculate_class_average_precision(
    class_scores: np.ndarray, class_targets: np.ndarray, max_k: int = None
) -> float:
    """
    Calculate the average precision at a given class.

    Args:
        class_scores (np.ndarray): Array of predicted class scores.
        class_targets (np.ndarray): Array of true class labels.
        max_k (int, optional): Maximum number of examples to consider. Defaults to None.

    Returns:
        float: Average precision at the given threshold.

    Raises:
        AssertionError: If the input and targets do not have the same shape.
        AssertionError: If the input is not 1-dimensional.
    """
    assert (
        class_scores.shape == class_targets.shape
    ), "The input and targets do not have the same shape"
    assert (
        class_scores.ndim == 1
    ), "The input has dimension {}, but expected it to be 1D".format(class_scores.shape)

    # sort examples
    indices = np.argsort(class_scores, axis=0)[::-1]

    total_cases = np.sum(class_targets)

    if max_k == None:
        max_k = len(indices)

    # Computes prec@i
    pos_count = 0.0
    total_count = 0.0
    precision_at_i = 0.0

    for i in range(max_k):
        label = class_targets[indices[i]]
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

        ap[k] = calculate_class_average_precision(scores_k, targets_k)

    return np.mean(ap)


def calculate_micro_prf(
    n_g: np.ndarray, n_p: np.ndarray, n_tp: np.ndarray
) -> Tuple[float]:
    """
    Calculate the micro-averaged precision, recall, F1 score, and F2 score.

    Parameters:
    - n_g (np.ndarray): Array containing the number of ground truth samples for each class.
    - n_p (np.ndarray): Array containing the number of predicted samples for each class.
    - n_tp (np.ndarray): Array containing the number of true positive samples for each class.

    Returns:
    - micro_p (float): Micro-averaged precision.
    - micro_r (float): Micro-averaged recall.
    - micro_f1 (float): Micro-averaged F1 score.
    - micro_f2 (float): Micro-averaged F2 score.
    """
    micro_p = precision(np.sum(n_tp), np.sum(n_p))
    micro_r = recall(np.sum(n_tp), np.sum(n_g))
    micro_f1 = fbeta_from_pr(micro_p, micro_r, 1)
    micro_f2 = fbeta_from_pr(micro_p, micro_r, 2)

    return micro_p, micro_r, micro_f1, micro_f2


def calculate_prf(
    n_g: Union[float, np.ndarray],
    n_p: Union[float, np.ndarray],
    n_tp: Union[float, np.ndarray],
):
    """
    Calculate precision, recall, F1 score, and F2 score from n_g, n_p, and n_tp.
    Supports float or np.ndarray.

    Args:
        n_g (Union[float, np.ndarray]): Number of ground truth samples.
        n_p (Union[float, np.ndarray]): Number of predicted samples.
        n_tp (Union[float, np.ndarray]): Number of true positive samples.

    Returns:
        Tuple[float]: A tuple containing precision, recall, F1 score, and F2 score
    """
    p = precision(n_tp, n_p)
    r = recall(n_tp, n_g)
    f1 = fbeta_from_pr(p, r, 1)
    f2 = fbeta_from_pr(p, r, 2)

    return p, r, f1, f2


def calculate_macro_prf_from_class_prf(
    class_p: np.ndarray, class_r: np.ndarray, class_f1: np.ndarray, class_f2: np.ndarray
) -> Tuple[float]:
    """
    Calculate macro precision, recall, F-score from class precision, recall, F-score.

    Args:
        class_p (np.ndarray): Array of class precision values.
        class_r (np.ndarray): Array of class recall values.
        class_f1 (np.ndarray): Array of class F1-score values.
        class_f2 (np.ndarray): Array of class F2-score values.

    Returns:
        Tuple[float]: A tuple containing macro precision, recall, F1-score, and F2-score.
    """
    return class_p.mean(), class_r.mean(), class_f1.mean(), class_f2.mean()


def calculate_macro_prf(
    n_g: np.ndarray, n_p: np.ndarray, n_tp: np.ndarray
) -> Tuple[float]:
    """
    Calculate macro-averaged precision, recall, F1 score, and F2 score.

    Args:
        n_g (np.ndarray): Array containing the number of ground truth samples for each class.
        n_p (np.ndarray): Array containing the number of predicted samples for each class.
        n_tp (np.ndarray): Array containing the number of true positive samples for each class.

    Returns:
        macro_p (float): Macro-averaged precision.
        macro_r (float): Macro-averaged recall.
        macro_f1 (float): Macro-averaged F1 score.
        macro_f2 (float): Macro-averaged F2 score.
    """
    class_p, class_r, class_f1, class_f2 = calculate_prf(n_g, n_p, n_tp)
    macro_p, macro_r, macro_f1, macro_f2 = calculate_macro_prf_from_class_prf(
        class_p, class_r, class_f1, class_f2
    )
    return macro_p, macro_r, macro_f1, macro_f2


def calculate_ciw_f2_from_class_f2(class_f2: np.ndarray, weights: List[float]) -> float:
    """
    Calculates the weighted average of class F2 scores to obtain the Class-Imbalanced Weighted F2 score.

    Args:
        class_f2 (np.ndarray): Array of class F2 scores.
        weights (List[float]): List of weights for each class.

    Returns:
        float: The Class-Imbalanced Weighted F2 score.
    """
    ciw_f2 = class_f2 * weights
    ciw_f2 = np.sum(ciw_f2) / np.sum(weights)

    return ciw_f2


def calculate_and_report_results(
    scores: np.ndarray,
    targets: np.ndarray,
    class_weights: List[float],
    threshold: Union[float, np.ndarray] = 0.5,
):
    """
    Calculates and reports various evaluation metrics based on the input scores and targets.

    Args:
        scores (np.ndarray): The predicted scores or probabilities for each class.
        targets (np.ndarray): The true labels or targets.
        class_weights (List[float]): The weights assigned to each class.
        threshold (Union[float, np.ndarray], optional): The threshold value(s) for classification. Defaults to 0.5.

    Returns:
        Tuple[Dict[str, float], Dict[str, float], Dict[str, List[float]]]: A tuple containing three dictionaries:
            - main_metrics: such as micro precision, recall, F1, F2, macro precision, recall, F1, F2, CIW F2, and mAP.
            - meta_metrics: defect/normal evaluation metrics such as defect/normal precision, recall, F, defect/normal counts.
            - class_metrics: class-wise metrics such as precision, recall, F1, F2, and class counts.
    """
    assert (
        scores.shape == targets.shape
    ), "The input and targets do not have the same size: Input: {} - Targets: {}".format(
        scores.shape, targets.shape
    )

    n_tp, n_p, n_g = calculate_class_wise_counts(scores, targets, threshold)

    # Micro Precision, Recall and F
    micro_p, micro_r, micro_f1, micro_f2 = calculate_micro_prf(n_g, n_p, n_tp)

    # Per-Class Precision, Recall and F
    class_p, class_r, class_f1, class_f2 = calculate_prf(n_g, n_p, n_tp)

    # Macro Precision, Recall and F
    macro_p, macro_r, macro_f1, macro_f2 = calculate_macro_prf_from_class_prf(
        class_p, class_r, class_f1, class_f2
    )

    # CIW F2
    ciw_f2 = calculate_ciw_f2_from_class_f2(class_f2, class_weights)

    # Mean Average Precision (mAP)
    mAP = get_mean_average_precision(scores, targets)

    # Get values for "implict" normal and defect classes
    n_tp_defect, n_p_defect, n_g_defect, n_tp_normal, n_p_normal, n_g_normal = (
        calculate_defect_normal_counts(scores, targets, threshold)
    )

    # Defect Normal Metrics
    defect_p, defect_r, defect_f1, defect_f2 = calculate_prf(
        n_g_defect, n_p_defect, n_tp_defect
    )
    normal_p, normal_r, normal_f1, normal_f2 = calculate_prf(
        n_g_normal, n_p_normal, n_tp_normal
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
