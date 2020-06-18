import torch
import torch.nn.functional as F

INV_LOG2 = 0.693147


def balanced_binary_cross_entropy(logits, labels, mask, weights):
    weights = (logits.new(weights).view(-1, 1, 1) - 1) * labels.float() + 1.
    weights = weights * mask.unsqueeze(1).float()
    return F.binary_cross_entropy_with_logits(logits, labels.float(), weights)


def uncertainty_loss(x, mask):
    """
    Loss which maximizes the uncertainty in invalid regions of the image
    """
    labels = ~mask
    x = x[labels.unsqueeze(1).expand_as(x)]
    xp, xm = x, -x
    entropy = xp.sigmoid() * F.logsigmoid(xp) + xm.sigmoid() * F.logsigmoid(xm)
    return 1. + entropy.mean() / INV_LOG2


def kl_divergence_loss(mu, logvar):
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())