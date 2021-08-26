# Sensitivity.py
# Michael Alban
# 07/28/2021

# largely a simplified analogy of the PyTorch implementation of Specificity: 
# https://github.com/PyTorchLightning/metrics/blob/master/torchmetrics/classification/specificity.py

import torch

from typing import Any, Callable, Optional

import torchmetrics.classification.stat_scores as stat_scores


class BinarySensitivity(stat_scores.StatScores):
    """
    Computes Sensitivity, or true positive rate, TP/(TP + FN), where TP
    represents the number of true positives and FN represents the number of
    false negatives.

    Args:
        threshold: 
            Threshold probability value for transforming probability predictions
            to binary (0, 1) predictions, in the case of binary or multi-label inputs.
    """

    def __init__(self, threshold: float = 0.5) -> None:
        super().__init__(threshold=threshold, multiclass=False)

    def compute(self) -> torch.Tensor:
        """
        Computes the sensitivity score based on inputs passed in to update previously.
        Return:
            
        """
        tp, fp, tn, fn = self._get_final_stats()
        if not tp+fn:
            return 0
        else:
            return tp/(tp + fn)

    @property
    def is_differentiable(self) -> bool:
        return False
