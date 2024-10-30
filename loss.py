import torch
import torch.nn.functional as F
from typing import Optional, Literal


class HybridLoss(torch.nn.Module):
    def __init__(
        self,
        class_counts: Optional[torch.Tensor] = None,
        defect_count: Optional[torch.Tensor] = None,
        beta: float = 0.9999,
        base_loss: Literal["sigmoid", "focal"] = "focal",
        focal_gamma: float = 2.0,
        meta_loss_weight: float = 1.0,
        push_mode: Literal["positive_push", "all_push"] = "positive_push",
    ):
        super().__init__()
        self.defect_class_weights = self._get_class_weights(class_counts, beta)
        self.base_loss = base_loss
        self.focal_gamma = focal_gamma
        self.meta_loss_weight = meta_loss_weight
        self.push_mode = push_mode
        
        if base_loss not in ["sigmoid", "focal"]:
            raise ValueError(f"Invalid base_loss '{base_loss}'")
        if push_mode not in ["positive_push", "all_push"]:
            raise ValueError(f"Invalid push_mode '{push_mode}'")

    @staticmethod
    def _focal_loss(
        logits: torch.Tensor,
        targets: torch.Tensor,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
    ) -> float:
        """
        Calculate the focal loss for a batch of predictions and targets.
        Focal loss = -alpha_t * (1-pt)^gamma * log(pt)
        pt is the probability of being classified to the true class.
        pt = p (if true class), otherwise pt = 1 - p. p = sigmoid(logit).
        Args:
            logits (torch.Tensor): logit scores for each class (batch_size, num_classes).
            targets (torch.Tensor): ground truth binary labels for each class (batch_size, num_classes).
            alpha (torch.Tensor): weights for each class to handle class imbalance (num_classes).
            gamma (float): A hyperparameter that controls the weighting scheme.
        Returns:
            float: focal_loss: A float32 scalar representing normalized total loss..
        """
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

        if gamma == 0.0:
            modulator = 1.0
        else:
            modulator = torch.exp(
                -gamma * targets * logits
                - gamma * torch.log(1 + torch.exp(-1.0 * logits))
            )

        loss = modulator * bce_loss

        if alpha is not None:
            weighted_loss = alpha * loss
            focal_loss = torch.sum(weighted_loss)
        else:
            focal_loss = torch.sum(loss)

        focal_loss /= (torch.sum(targets) + 1e-6)
        
        return focal_loss

    @staticmethod
    def _get_class_weights(class_counts: torch.Tensor, beta: float) -> torch.Tensor:
        """
        Calculate class weights using the effective number of samples method.
        Args:
            class_counts (torch.Tensor): the count of samples for each class.
            beta (float): A hyperparameter that controls the weighting scheme.
        Returns:
            torch.Tensor: the calculated weights for each class.
        """
        effective_num = 1.0 - torch.pow(beta, class_counts)
        weights = (1.0 - beta) / effective_num
        weights = weights / torch.sum(weights) * len(class_counts)

        return weights

    @staticmethod
    def _calculate_batch_balancing_weights(
        class_weights: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate balancing weights for a batch of targets based on class weights.
        Args:
            class_weights (torch.Tensor): (num_classes,) the weights for each class.
            targets (torch.Tensor): (batch_size, num_classes) the one-hot encoded target classes.
        Returns:
            torch.Tensor: (batch_size, num_classes) balancing weights for each sample in the batch.
        """
        weights = class_weights.to(targets.device).float()
        weights = weights.unsqueeze(0)  # (1, num_classes)
        weights = weights.expand(targets.shape[0], -1)  # (batch_size, num_classes)
        weights = torch.sum(weights * targets, 1, keepdim=True)  # (batch_size, 1)
        weights = weights.expand(-1, targets.shape[1])  # (batch_size, num_classes)

        return weights

    def _calculate_multi_label_loss(
        self, logits: torch.Tensor, targets: torch.Tensor, class_weights: torch.Tensor
    ) -> float:
        """
        Calculate the multi-label loss for a batch of predictions and targets.
        Args:
            logits (torch.Tensor): logit scores for each class (batch_size, num_classes).
            targets (torch.Tensor): ground truth binary labels for each class (batch_size, num_classes).
            class_weights (torch.Tensor): weights for each class to handle class imbalance (num_classes).
        Returns:
            float: The mean multi-label loss for the batch.
        """
        weights = self._calculate_batch_balancing_weights(class_weights, targets)
        
        if self.base_loss == "sigmoid":
            defect_type_loss = F.binary_cross_entropy_with_logits(
                logits, targets, weights, reduction="none"
            ) 
            defect_type_loss = torch.sum(defect_type_loss) / (torch.sum(targets) + 1e-6)
        else:
            defect_type_loss = self._focal_loss(
                logits, targets, weights, self.focal_gamma
            )

        return defect_type_loss

    def _calculate_positive_push_meta_loss(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> float:
        """
        Calculate the target-push meta loss for defect detection.
        the meta positive node consists only of the nodes for the defect that are present in the image.

        Args:
            logits (torch.Tensor): The predicted logits from the model with shape (batch_size, num_classes).
            targets (torch.Tensor): The ground truth targets with shape (batch_size, num_classes).
        Returns:
            float: Average defect loss for the batch
        """
        meta_targets = torch.sum(targets, 1, True).clamp(0, 1)  # (batch_size, 1) 0 or 1

        # copy target to a new tensor
        meta_logit_weights = targets.clone()
        # set all columns to 1 if the column is normal (no defect)
        meta_logit_weights[meta_targets.expand(-1, targets.shape[1]) == 0] = 1

        meta_logits = torch.sum(
            logits * meta_logit_weights, dim=1, keepdim=True
        ) / torch.sum(meta_logit_weights, dim=1, keepdim=True)

        if self.base_loss == "sigmoid":
            meta_loss = F.binary_cross_entropy_with_logits(
                meta_logits, meta_targets, reduction="mean"
            ) 
        else:
            meta_loss = self._focal_loss(
                meta_logits, meta_targets, gamma=self.focal_gamma
            )

        return meta_loss

    def _calculate_all_push_meta_loss(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> float:
        """
        Calculate the all-push meta loss for defect detection.
        the meta positive node consists of all the nodes in the image.

        Args:
            logits (torch.Tensor): logits from the model with shape (batch_size, num_classes).
            targets (torch.Tensor): ground truth targets with shape (batch_size, num_classes).
        Returns:
            float: Average defect loss for the batch
        """
        meta_targets = torch.sum(targets, 1, True).clamp(0, 1)  # (batch_size, 1) 0 or 1

        meta_logits = torch.mean(logits, dim=1, keepdim=True)

        if self.base_loss == "sigmoid":
            meta_loss = F.binary_cross_entropy_with_logits(
                meta_logits, meta_targets, reduction="mean"
            )
        else:
            meta_loss = self._focal_loss(
                meta_logits, meta_targets, gamma=self.focal_gamma
            )

        return meta_loss

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        defect_type_loss = self._calculate_multi_label_loss(
            logits, targets, self.defect_class_weights
        )

        if self.push_mode == "positive_push":
            meta_loss = self._calculate_positive_push_meta_loss(logits, targets)
        else:
            meta_loss = self._calculate_all_push_meta_loss(logits, targets)

        final_loss = defect_type_loss + self.meta_loss_weight * meta_loss

        return final_loss


if __name__ == "__main__":
    NUM_CLASSES = 5
    BETA = 0.9999
    BASE_LOSS = "focal"
    FOCAL_GAMMA = 2.0
    META_LOSS_WEIGHT = 0.1
    PUSH_MODE = "positive_push"
    CLASS_COUNTS = torch.Tensor([2, 3, 1, 2, 2])

    torch.manual_seed(0)

    logits = torch.rand(10, NUM_CLASSES).float()

    targets = torch.rand(10, NUM_CLASSES).float()
    targets[targets > 0.5] = 1.0
    targets[targets <= 0.5] = 0.0

    print("\nLogits: ", logits)
    print("\nLabels: ", targets)

    criterion = HybridLoss(
        class_counts=CLASS_COUNTS,
        beta=BETA,
        base_loss=BASE_LOSS,
        focal_gamma=FOCAL_GAMMA,
        meta_loss_weight=META_LOSS_WEIGHT,
        push_mode=PUSH_MODE,
    )

    loss = criterion(logits, targets)
    print("\nLoss: ", loss)
