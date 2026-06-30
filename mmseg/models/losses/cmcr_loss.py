import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES


@LOSSES.register_module()
class CMCRLoss(nn.Module):
    """Cross-view and modal consistency regularisation.

    The loss expects decoder-stage predictions from original, weak, and strong
    views. It combines mask consistency, query-class consistency, and optional
    pixel-text attention consistency.
    """

    def __init__(self,
                 mask_weight=1.0,
                 class_weight=1.0,
                 align_weight=1.0,
                 loss_weight=1.0):
        super().__init__()
        self.mask_weight = mask_weight
        self.class_weight = class_weight
        self.align_weight = align_weight
        self.loss_weight = loss_weight

    def forward(self,
                mask_preds,
                weak_mask_preds,
                strong_mask_preds,
                class_logits,
                weak_class_logits,
                strong_class_logits,
                attention_maps=None,
                weak_attention_maps=None,
                strong_attention_maps=None):
        loss = self.mask_weight * self._mask_loss(mask_preds, weak_mask_preds, strong_mask_preds)
        loss = loss + self.class_weight * self._class_loss(
            class_logits, weak_class_logits, strong_class_logits)
        if attention_maps is not None:
            loss = loss + self.align_weight * self._align_loss(
                attention_maps, weak_attention_maps, strong_attention_maps)
        return loss * self.loss_weight

    @staticmethod
    def _mask_loss(anchor, weak, strong):
        anchor_prob = anchor.detach().sigmoid()
        weak_loss = F.binary_cross_entropy_with_logits(weak, anchor_prob)
        strong_loss = F.binary_cross_entropy_with_logits(strong, anchor_prob)
        return 0.5 * (weak_loss + strong_loss)

    @staticmethod
    def _class_loss(anchor, weak, strong):
        anchor_prob = anchor.detach().softmax(dim=-1)
        weak_log = F.log_softmax(weak, dim=-1)
        strong_log = F.log_softmax(strong, dim=-1)
        weak_loss = F.kl_div(weak_log, anchor_prob, reduction='batchmean')
        strong_loss = F.kl_div(strong_log, anchor_prob, reduction='batchmean')
        return 0.5 * (weak_loss + strong_loss)

    @staticmethod
    def _align_loss(anchor, weak, strong):
        anchor_prob = anchor.detach()
        weak_loss = F.mse_loss(weak, anchor_prob)
        strong_loss = F.mse_loss(strong, anchor_prob)
        return 0.5 * (weak_loss + strong_loss)
