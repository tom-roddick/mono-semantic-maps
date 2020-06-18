import torch
import torch.nn as nn
from ..nn.losses import balanced_binary_cross_entropy, uncertainty_loss, \
    kl_divergence_loss

class OccupancyCriterion(nn.Module):

    def __init__(self, xent_weight=1., uncert_weight=0., class_weights=None):
        super().__init__()

        self.xent_weight = xent_weight
        self.uncert_weight = uncert_weight

        if class_weights is None:
            self.class_weights = torch.ones(1)
        else:
            self.class_weights = torch.tensor(class_weights)
    

    def forward(self, logits, labels, mask, *args):

        # Compute binary cross entropy loss
        self.class_weights = self.class_weights.to(logits)
        bce_loss = balanced_binary_cross_entropy(
            logits, labels, mask, self.class_weights)
        
        # Compute uncertainty loss for unknown image regions
        uncert_loss = uncertainty_loss(logits, mask)

        return bce_loss * self.xent_weight + uncert_loss * self.uncert_weight



class VaeOccupancyCriterion(OccupancyCriterion):

    def __init__(self, xent_weight=0.9, uncert_weight=0., kld_weight=0.1, 
                 class_weights=None):
        super().__init__(xent_weight, uncert_weight, class_weights)

        self.kld_weight = kld_weight
    
    def forward(self, logits, labels, mask, mu, logvar):

        kld_loss = kl_divergence_loss(mu, logvar)
        occ_loss = super().forward(logits, labels, mask)
        return occ_loss + kld_loss * self.kld_weight