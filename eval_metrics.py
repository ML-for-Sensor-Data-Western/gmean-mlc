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
