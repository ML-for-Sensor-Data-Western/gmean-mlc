from typing import Literal, Optional

import torch
import torch.nn.functional as F
from torchvision.ops import sigmoid_focal_loss


class HybridLoss(torch.nn.Module):
    def __init__(
        self,
        class_counts: Optional[torch.Tensor] = None,
        normal_count: Optional[float] = None,
        class_balancing_beta: float = 0.9999,
        base_loss: Literal["bce", "focal"] = "focal",
        focal_gamma: float = 2.0,
        meta_loss_weight: float = 1.0,
        meta_loss_beta: float = 0.1,
    ):
        super().__init__()
        self.defect_class_weights, self.normal_weight = self._get_class_weights(
            class_counts, normal_count, class_balancing_beta
        )
        self.base_loss = base_loss
        self.focal_gamma = focal_gamma
        self.meta_loss_weight = meta_loss_weight
        self.meta_loss_beta = meta_loss_beta

        if base_loss not in ["bce", "focal"]:
            raise ValueError(f"Invalid base_loss '{base_loss}'")

    @staticmethod
    def _calculate_stable_focal_loss(
        logits: torch.Tensor,
        targets: torch.Tensor,
        alpha: float = -1,
        gamma: float = 2.0,
        reduction: Literal["sum", "mean", "none"] = "none",
    ) -> torch.Tensor:
        """
        Calculate the optimization-stable focal loss for a batch of predictions and targets.
        Ref: https://github.com/vandit15/Class-balanced-loss-pytorch/blob/master/class_balanced_loss.py
        Ref: https://github.com/fcakyon/balanced-loss/blob/main/balanced_loss/losses.py
        Args:
            logits (torch.Tensor): logit scores for each class (batch_size, num_classes).
            targets (torch.Tensor): ground truth binary labels for each class (batch_size, num_classes).
            alpha (float): Weighting factor in range (0,1) to balance positive vs negative examples or -1 for ignore. Default: ``-1`` We wont weight here.
            gamma (float): Exponent of the modulating factor (1 - p_t) to balance easy vs hard examples. Default: ``2``.
            reduction (string): ``'none'`` | ``'mean'`` | ``'sum'``
        Returns:
            torch.Tensor: Loss tensor with chosen reduction
        """
        ce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

        if gamma == 0:
            modulator = 1.0
        else:
            modulator = torch.exp(
                -gamma * targets * logits
                - gamma * torch.log(1 + torch.exp(-1.0 * logits))
            )

        loss = modulator * ce_loss

        if alpha >= 0:
            alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
            loss = alpha_t * loss

        # Check reduction option and return loss accordingly
        if reduction == "none":
            pass
        elif reduction == "mean":
            loss = loss.mean()
        elif reduction == "sum":
            loss = loss.sum()
        else:
            raise ValueError(
                f"Invalid Value for arg 'reduction': '{reduction} \n Supported reduction modes: 'none', 'mean', 'sum'"
            )
        return loss

    @staticmethod
    def _get_class_weights(
        class_counts: torch.Tensor, normal_count: float, beta: float
    ) -> torch.Tensor:
        """
        Calculate class weights using the effective number of samples method.
        Args:
            class_counts (torch.Tensor): the count of samples for each class.
            normal_count (float): the count of samples for the normal class.
            beta (float): A hyperparameter that controls the weighting scheme.
        Returns:
            torch.Tensor: the calculated weights for each class.
            float: the calculated weight for the normal class.
        """
        all_class_counts = torch.cat([class_counts, torch.tensor([normal_count])])
        effective_num = 1.0 - torch.pow(beta, all_class_counts)
        all_weights = (1.0 - beta) / effective_num
        all_weights = all_weights / torch.sum(all_weights) * len(all_class_counts)

        class_weights = all_weights[:-1]
        normal_weight = all_weights[-1]

        return class_weights, normal_weight

    @staticmethod
    def _calculate_batch_balancing_weights(
        class_weights: torch.Tensor, normal_weight: float, targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate balancing weights for a batch of targets based on class weights.
        Args:
            class_weights (torch.Tensor): (num_classes,) the weights for each class.
            targets (torch.Tensor): (batch_size, num_classes) the one-hot encoded target classes.
        Returns:
            torch.Tensor: (batch_size, 1) balancing weights for each sample in the batch.
        """
        weights = class_weights.type_as(targets).float()
        weights = weights.unsqueeze(0)  # (1, num_classes)
        weights = weights.expand(targets.shape[0], -1)  # (batch_size, num_classes)
        weights = torch.sum(weights * targets, 1, keepdim=True)  # (batch_size, 1)
        weights[weights == 0] = normal_weight  # set normal class weight

        return weights

    def _calculate_multi_label_loss(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate the multi-label loss for a batch of predictions and targets.
        Args:
            logits (torch.Tensor): logit scores for each class (batch_size, num_classes).
            targets (torch.Tensor): ground truth binary labels for each class (batch_size, num_classes).
        Returns:
            torch.Tensor (BS, 1): element-wise multi-label loss for the batch.
        """
        if self.base_loss == "bce":
            defect_type_loss = F.binary_cross_entropy_with_logits(
                logits, targets, reduction="none"
            )
        else:
            defect_type_loss = sigmoid_focal_loss(
            # defect_type_loss = self._calculate_stable_focal_loss(
                logits, targets, alpha=-1, gamma=self.focal_gamma, reduction="none"
            )

        return defect_type_loss
    
    def _calculate_aux_loss2(
        self, logits: torch.Tensor, targets: torch.Tensor
    ):
        aux_targets = torch.ones(targets.shape[0], 1).type_as(logits)
        aux_logits = torch.sum((-1)**(1-targets)*logits, 1, keepdim=True)/targets.shape[1]
        
        if self.base_loss == "bce":
            aux_loss = F.binary_cross_entropy_with_logits(
                aux_logits, aux_targets, reduction="none"
            )
        else:
            aux_loss = sigmoid_focal_loss(
            # aux_loss = self._calculate_stable_focal_loss(
                aux_logits,
                aux_targets,
                alpha=-1,
                gamma=self.focal_gamma,
                reduction="none",
            )
        return aux_loss
    

    def _calculate_aux_loss(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate the meta loss for defect detection.
        the meta node consists of weighted nodes for the defect that are present in the image,
        or all nodes if the image is normal (no defect).

        Args:
            logits (torch.Tensor): The predicted logits from the model with shape (batch_size, num_classes).
            targets (torch.Tensor): The ground truth targets with shape (batch_size, num_classes).
        Returns:
            Tensor (BS, 1): element-wise defect loss for the batch
        """
        meta_targets = torch.sum(targets, 1, True).clamp(0, 1)  # (batch_size, 1) 0 or 1

        # copy target to a new tensor
        meta_logit_weights = targets.clone()
        # set all columns to 1 if the column is normal (no defect)
        meta_logit_weights[meta_targets.expand(-1, targets.shape[1]) == 0] = 1
        # set remaining 0s to beta
        meta_logit_weights[meta_logit_weights == 0] = self.meta_loss_beta

        meta_logits = torch.sum(
            logits * meta_logit_weights, dim=1, keepdim=True
        ) / torch.sum(meta_logit_weights, dim=1, keepdim=True)

        if self.base_loss == "bce":
            meta_loss = F.binary_cross_entropy_with_logits(
                meta_logits, meta_targets, reduction="none"
            )
        else:
            meta_loss = sigmoid_focal_loss(
            # meta_loss = self._calculate_stable_focal_loss(
                meta_logits,
                meta_targets,
                alpha=-1,
                gamma=self.focal_gamma,
                reduction="none",
            )

        return meta_loss

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        defect_type_loss = self._calculate_multi_label_loss(logits, targets)
        defect_type_loss = torch.sum(defect_type_loss, 1, keepdim=True)

        meta_loss = self._calculate_aux_loss(logits, targets)
        # meta_loss = self._calculate_aux_loss2(logits, targets)

        final_loss = defect_type_loss + self.meta_loss_weight * meta_loss

        balancing_weights = self._calculate_batch_balancing_weights(
            self.defect_class_weights, self.normal_weight, targets
        )  # (bs, 1)

        normal_target = 1 - torch.sum(targets, 1, True).clamp(0, 1)  # (bs, 1)
        # loss_denominator = torch.sum(targets, 1, keepdim=True) + normal_target
        # final_loss = torch.mean(
        #     final_loss * balancing_weights / loss_denominator
        # )
        loss_denominator = torch.sum(targets) + torch.sum(normal_target)
        final_loss = torch.sum(final_loss * balancing_weights) / loss_denominator

        return final_loss


if __name__ == "__main__":
    NUM_CLASSES = 5
    CLASS_BALANCING_BETA = 0.9999
    BASE_LOSS = "focal"
    FOCAL_GAMMA = 2.0
    META_LOSS_WEIGHT = 0.1
    META_LOSS_BETA = 0.1
    CLASS_COUNTS = torch.Tensor([2, 3, 1, 2, 2])
    NORMAL_COUNT = 5

    torch.manual_seed(0)

    logits = torch.rand(10, NUM_CLASSES).float()

    targets = torch.rand(10, NUM_CLASSES).float()
    targets[targets > 0.5] = 1.0
    targets[targets <= 0.5] = 0.0

    print("\nLogits: ", logits)
    print("\nLabels: ", targets)

    criterion = HybridLoss(
        class_counts=CLASS_COUNTS,
        normal_count=NORMAL_COUNT,
        class_balancing_beta=CLASS_BALANCING_BETA,
        base_loss=BASE_LOSS,
        focal_gamma=FOCAL_GAMMA,
        meta_loss_weight=META_LOSS_WEIGHT,
        meta_loss_beta=META_LOSS_BETA,
    )

    loss = criterion(logits, targets)
    print("\nLoss: ", loss)
