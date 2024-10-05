from torchmetrics import Metric
from torchmetrics.utilities import dim_zero_cat
import torch

class CustomMultiLabelAveragePrecision(Metric):
    
    def __init__(self, num_labels, **kwargs):
        super().__init__(**kwargs)
        self.num_labels = num_labels
        # States for accumulating predictions and targets
        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("target", default=[], dist_reduce_fx="cat")

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        # Accumulate predictions and targets for each batch
        self.preds.append(preds)
        self.target.append(target)

    def compute(self) -> torch.Tensor:
        # Concatenate all accumulated predictions and targets
        all_preds = dim_zero_cat(self.preds)
        all_targets = dim_zero_cat(self.target)

        ap_per_class = torch.zeros(self.num_labels)

        # Go through each class and calculate AP
        for class_idx in range(self.num_labels):
            class_preds = all_preds[:, class_idx]
            class_target = all_targets[:, class_idx]

            # Sort by confidence
            sorted_indices = torch.argsort(class_preds, descending=True)

            # Total number of positive cases for this class
            total_cases = torch.sum(class_target)

            # Initialize counters
            positive_count = 0.0
            total_count = 0.0
            precision_sum = 0.0

            # Go through each prediction and compute precision and recall
            for i in sorted_indices:
                label = class_target[i]
                total_count += 1

                if label == 1:
                    positive_count += 1
                    precision_sum += positive_count / total_count
                if positive_count == total_cases:
                    break
            if positive_count > 0:
                precision_sum /= positive_count
            else:
                precision_sum = 0.0
            
            # Store AP for this class
            ap_per_class[class_idx] = precision_sum

        # Return the mean AP across all classes
        return ap_per_class.mean()


class MaxMultiLabelFbetaScore(Metric):
    """Calculate the maximum Fbeta score for each class across all thresholds."""
    
    def __init__(self, num_labels, beta, **kwargs):
        super().__init__(**kwargs)
        self.num_labels = num_labels
        self.beta = beta
        # States for accumulating predictions and targets
        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("target", default=[], dist_reduce_fx="cat")

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        # Accumulate predictions and targets for each batch
        self.preds.append(preds)
        self.target.append(target)

    def compute(self) -> torch.Tensor:
        # Concatenate all accumulated predictions and targets
        all_preds = dim_zero_cat(self.preds)
        all_targets = dim_zero_cat(self.target)

        thresholds = [t/100 for t in range(5, 100, 5)]
        f_per_class_per_t = torch.zeros(len(thresholds), self.num_labels)
        
        n_g = torch.sum(all_targets, dim=0)

        for i, threshold in enumerate(thresholds):
            n_p = torch.sum(all_preds >= threshold, dim=0)
            n_tp = torch.sum((all_preds >= threshold) * all_targets, dim=0)
        
            fbeta = self.fbeta_score(n_g, n_p, n_tp, self.beta)
            f_per_class_per_t[i] = fbeta
        
        max_f_per_class = torch.max(f_per_class_per_t, dim=0).values          

        # Return the mean F1 score across all classes
        return max_f_per_class.mean()
    
    @staticmethod
    def fbeta_score(n_g: torch.Tensor, n_p: torch.Tensor, n_tp: torch.Tensor, beta: float) -> torch.Tensor:
        p = n_tp / n_p + 1e-7
        r = n_tp / n_g + 1e-7
        fbeta = (1 + beta ** 2) * p * r / (beta ** 2 * p + r)
        return fbeta