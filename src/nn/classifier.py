import math
import torch.nn as nn
import torch.nn.functional as F

class LinearClassifier(nn.Conv2d):

    def __init__(self, in_channels, num_class):
        super().__init__(in_channels, num_class, 1)
    
    def initialise(self, prior):
        self.weight.data.zero_()
        self.bias.data.fill_(-math.log((1 - prior) / prior))
    


class BayesianClassifier(nn.Module):

    def __init__(self, in_channels, num_class, num_samples=40):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, num_class, 1)
        self.num_samples = num_samples
    
    def initialise(self, prior):
        self.conv.weight.data.zero_()
        self.conv.bias.data.fill_(-math.log((1 - prior) / prior))
    
    def forward(self, features):

        if self.training:
            # At training time, apply dropout once
            features = F.dropout2d(features, 0.5, training=True)
            logits = self.conv(features)

        else:
            # At test time, apply dropout multiple times and average the result
            mean_score = 0
            for _ in self.num_samples:
                drop_feats = F.dropout2d(features, 0.5, training=True)
                mean_score += F.sigmoid(self.classifier(drop_feats))
            mean_score = mean_score / self.num_samples

            # Convert back into logits format
            logits = torch.log(mean_score) - torch.log1p(-mean_score)
        
        return logits


