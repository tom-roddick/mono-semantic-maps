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


def prior_uncertainty_loss(x, mask, priors):
    priors = x.new(priors).view(1, -1, 1, 1).expand_as(x)
    xent = F.binary_cross_entropy_with_logits(x, priors, reduce=False)
    return (xent * (~mask).float().unsqueeze(1)).mean() 


def kl_divergence_loss(mu, logvar):
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())


def focal_loss(logits, labels, mask, alpha=0.5, gamma=2):
    
    bce_loss = F.binary_cross_entropy_with_logits(logits, labels.float(), 
                                                  reduce=False)
    pt = torch.exp(-bce_loss)
    at = pt.new([alpha, 1 - alpha])[labels.long()]
    focal_loss = at * (1 - pt) ** gamma * bce_loss

    return (focal_loss * mask.unsqueeze(1).float()).mean()


def prior_offset_loss(logits, labels, mask, priors):

    priors = logits.new(priors).view(-1, 1, 1)
    prior_logits = torch.log(priors / (1 - priors))
    labels = labels.float()

    weights = .5 / priors * labels + .5 / (1 - priors) * (1 - labels)
    weights = weights * mask.unsqueeze(1).float()
    return F.binary_cross_entropy_with_logits(logits - prior_logits, labels, 
                                              weights)
