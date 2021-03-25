import torch
import torch.nn as nn
from ..nn.losses import balanced_binary_cross_entropy, uncertainty_loss, \
    kl_divergence_loss, focal_loss, prior_offset_loss, prior_uncertainty_loss

# class OccupancyCriterion(nn.Module):

#     def __init__(self, xent_weight=1., uncert_weight=0., class_weights=None):
#         super().__init__()

#         self.xent_weight = xent_weight
#         self.uncert_weight = uncert_weight

#         if class_weights is None:
#             self.class_weights = torch.ones(1)
#         else:
#             self.class_weights = torch.tensor(class_weights)
    

#     def forward(self, logits, labels, mask, *args):

#         # Compute binary cross entropy loss
#         self.class_weights = self.class_weights.to(logits)
#         bce_loss = balanced_binary_cross_entropy(
#             logits, labels, mask, self.class_weights)
        
#         # Compute uncertainty loss for unknown image regions
#         uncert_loss = uncertainty_loss(logits, mask)

#         return bce_loss * self.xent_weight + uncert_loss * self.uncert_weight


class OccupancyCriterion(nn.Module):

    def __init__(self, priors, xent_weight=1., uncert_weight=0., 
                 weight_mode='sqrt_inverse'):
        super().__init__()

        self.xent_weight = xent_weight
        self.uncert_weight = uncert_weight

        self.priors = torch.tensor(priors)

        if weight_mode == 'inverse':
            self.class_weights = 1 / self.priors
        elif weight_mode == 'sqrt_inverse':
            self.class_weights = torch.sqrt(1 / self.priors)
        elif weight_mode == 'equal':
            self.class_weights = torch.ones_like(self.priors)
        else:
            raise ValueError('Unknown weight mode option: ' + weight_mode)
    

    def forward(self, logits, labels, mask, *args):

        # Compute binary cross entropy loss
        self.class_weights = self.class_weights.to(logits)
        bce_loss = balanced_binary_cross_entropy(
            logits, labels, mask, self.class_weights)
        
        # Compute uncertainty loss for unknown image regions
        self.priors = self.priors.to(logits)
        uncert_loss = prior_uncertainty_loss(logits, mask, self.priors)

        return bce_loss * self.xent_weight + uncert_loss * self.uncert_weight



class FocalLossCriterion(nn.Module):

    def __init__(self, alpha, gamma):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, logits, labels, mask, *args):
        return focal_loss(logits, labels, mask, self.alpha, self.gamma)


class PriorOffsetCriterion(nn.Module):

    def __init__(self, priors):
        super().__init__()
        self.priors = priors
    
    def forward(self, logits, labels, mask, *args):
        return prior_offset_loss(logits, labels, mask, self.priors)




class VaeOccupancyCriterion(OccupancyCriterion):

    def __init__(self, priors, xent_weight=0.9, uncert_weight=0., weight_mode='sqrt_inverse',  kld_weight=0.1):
        super().__init__(priors, xent_weight, uncert_weight, weight_mode)

        self.kld_weight = kld_weight
    
    def forward(self, logits, labels, mask, mu, logvar):

        kld_loss = kl_divergence_loss(mu, logvar)
        occ_loss = super().forward(logits, labels, mask)
        return occ_loss + kld_loss * self.kld_weight
