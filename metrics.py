import numpy as np

# False Positives = n_p - n_tp
# False Negatives = n_g - n_tp
# True Positives = n_tp
# True Negatives = n_examples - n_p + (n_g - n_tp)


def get_class_counts(scores, targets, threshold):
    _, n_class = scores.shape

    # Arrays to hold binary classification information, size n_class +1 to also hold the implicit normal class
    n_tp = np.zeros(n_class)  # True positives
    n_p = np.zeros(n_class)  # Total Positives
    n_g = np.zeros(n_class)  # Total number of Ground Truth occurences

    # Array to hold the average precision metric.
    ap = np.zeros(n_class)

    for k in range(n_class):
        scores_k = scores[:, k]
        targets_k = targets[:, k]
        # Necessary if using MultiLabelSoftMarginLoss, instead of BCEWithLogitsLoss
        targets_k[targets_k == -1] = 0

        n_g[k] = np.sum(targets_k == 1)
        n_p[k] = np.sum(scores_k >= threshold)
        n_tp[k] = np.sum(targets_k * (scores_k >= threshold))

        ap[k] = get_average_precision(scores_k, targets_k)

    # If Np is 0 for any class, set to 1 to avoid division with 0
    n_p[n_p == 0] = 1

    return n_tp, n_p, n_g, ap


def get_defect_normal_counts(scores, targets, threshold):
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


def get_mean_average_precision(ap):
    return np.mean(ap)


def get_exact_match_accuracy(scores, targets, threshold=0.5):
    n_examples, n_class = scores.shape

    binary_mat = np.equal(targets, (scores >= threshold))
    row_sums = binary_mat.sum(axis=1)

    perfect_match = np.zeros(row_sums.shape)
    perfect_match[row_sums == n_class] = 1

    EMAcc = np.sum(perfect_match) / n_examples

    return EMAcc


def evaluation(scores, targets, weights, threshold=0.5):
    assert (
        scores.shape == targets.shape
    ), "The input and targets do not have the same size: Input: {} - Targets: {}".format(
        scores.shape, targets.shape
    )

    n_tp, n_p, n_g, ap = get_class_counts(scores, targets, threshold)

    # Micro Precision, Recall and F1
    micro_p, micro_r, micro_f1, micro_f2 = get_micro_metrics(n_g, n_p, n_tp)

    # Per-Class Precision, Recall and F1
    class_p, class_r, class_f1, class_f2 = get_class_metrics(n_g, n_p, n_tp)

    # Macro metrics
    macro_p, macro_r, macro_f1, macro_f2 = get_macro_metrics(
        class_p, class_r, class_f1, class_f2
    )

    ciw_f2 = get_class_weighted_f2(class_f2, weights)

    # Zero-One exact match accuracy
    EMAcc = get_exact_match_accuracy(scores, targets, threshold)

    # Mean Average Precision (mAP)
    mAP = get_mean_average_precision(ap)

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
        "EMAcc": EMAcc,
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
        "DEFECT_NP": n_p_defect,
        "DEFECT_NTP": n_tp_defect,
        "DEFECT_NG": n_g_defect,
        "NORMAL_NP": n_p_normal,
        "NORMAL_NTP": n_tp_normal,
        "NORMAL_NG": n_g_normal,
    }

    # Class-wise Counts and Metrics
    class_metrics = {
        "P_CLS": list(class_p),
        "R_CLS": list(class_r),
        "F1_CLS": list(class_f1),
        "F2_CLS": list(class_f2),
        "AP": list(ap),
        "NP": list(n_p),
        "NTP": list(n_tp),
        "NG": list(n_g),
    }

    return main_metrics, meta_metrics, class_metrics
