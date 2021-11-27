import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PyramidOccupancyNetwork(nn.Module):


    def __init__(self, frontend, transformer, topdown, classifier, ipm_transform):
        super().__init__()


        self.frontend = frontend
        self.transformer = transformer
        self.topdown = topdown
        self.classifier = classifier
        self.ipm_transform = ipm_transform
    

    def forward(self, image, calib, *args):

        # Extract multiscale feature maps
        feature_maps = self.frontend(image)

        # Transform image features to birds-eye-view
        bev_feats = self.transformer(feature_maps, calib)
        
        ipm_transform_on = True
        if ipm_transform_on:
            # concatenate the ipm features
            import pdb;pdb.set_trace()
            ipm = self.ipm_transform(image)
            bev_feats = torch.cat((ipm, bev_feats), 1)
        
        
        # Apply topdown network
        td_feats = self.topdown(bev_feats)

        # Predict individual class log-probabilities
        logits = self.classifier(td_feats)
        return logits