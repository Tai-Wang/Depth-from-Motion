# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.ops import diff_iou_rotated_3d

from mmdet.models.losses.utils import weighted_loss
from ..builder import LOSSES


@weighted_loss
def iou3d_loss(pred, target):
    """Smooth L1 loss with uncertainty.

    Args:
        pred (torch.Tensor): Predicted boxes of shape (N, 7).
        target (torch.Tensor): Target boxes of shape (N, 7).

    Returns:
        torch.Tensor: Calculated loss
    """
    assert target.numel() > 0
    pred = pred.contiguous()
    target = target.contiguous()
    target = torch.where(torch.isnan(target), pred,
                         target)  # ignore nan targets

    if pred.size(0) > 0:
        loss = 1 - diff_iou_rotated_3d(pred.unsqueeze(0), target.unsqueeze(0))
    else:
        loss = (pred - target).sum(1) * 0.

    return loss


@LOSSES.register_module()
class IOU3DLoss(nn.Module):
    """IOU3DLoss.

    Args:
        reduction (str, optional): The method to reduce the loss.
            Options are 'none', 'mean' and 'sum'. Defaults to 'mean'.
        loss_weight (float, optional): The weight of loss. Defaults to 1.0
    """

    def __init__(self, reduction='mean', loss_weight=1.0):
        super(IOU3DLoss, self).__init__()
        assert reduction in ['none', 'sum', 'mean']
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning target of the prediction.
            sigma (torch.Tensor): The sigma for uncertainty.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss_bbox = self.loss_weight * iou3d_loss(
            pred,
            target,
            weight,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        return loss_bbox
