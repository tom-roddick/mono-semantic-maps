import math
from operator import mul
from functools import reduce
import torch.nn as nn

from .pyramid import PyramidOccupancyNetwork
from .ved import VariationalEncoderDecoder
from .vpn import VPNModel
from .criterion import OccupancyCriterion, VaeOccupancyCriterion, \
    FocalLossCriterion, PriorOffsetCriterion

from ..nn.fpn import FPN50
from ..nn.topdown import TopdownNetwork
from ..nn.pyramid import TransformerPyramid
from ..nn.classifier import LinearClassifier, BayesianClassifier



def build_model(model_name, config):

    if model_name == 'pyramid':
        model = build_pyramid_occupancy_network(config)
    elif model_name == 'ved':
        model = build_variational_encoder_decoder(config)
    elif model_name == 'vpn':
        model = build_view_parsing_network(config)
    else:
        raise ValueError("Unknown model name '{}'".format(model_name))
    
    if len(config.gpus) > 1:
        model = nn.DataParallel(model.cuda(), config.gpus)
    elif len(config.gpus) == 1:
        model.cuda()
    
    return model


def build_criterion(model_name, config):

    if model_name == 'ved':
        criterion = VaeOccupancyCriterion(config.prior,
                                          config.xent_weight, 
                                          config.uncert_weight,
                                          config.weight_mode,
                                          config.kld_weight, 
                                          )
                                          
    elif config.loss_fn == 'focal':
        criterion = FocalLossCriterion(config.focal.alpha, config.focal.gamma)
    elif config.loss_fn == 'prior':
        criterion = PriorOffsetCriterion(config.prior)
    else:
        criterion = OccupancyCriterion(config.prior, config.xent_weight, 
                                       config.uncert_weight, config.weight_mode)
    
    if len(config.gpus) > 0:
        criterion.cuda()
    
    return criterion



def build_pyramid_occupancy_network(config):

    # Build frontend
    frontend = FPN50()

    # Build transformer pyramid
    tfm_resolution = config.map_resolution * reduce(mul, config.topdown.strides)
    transformer = TransformerPyramid(256, config.tfm_channels, tfm_resolution,
                                     config.map_extents, config.ymin, 
                                     config.ymax, config.focal_length)

    # Build topdown network
    topdown = TopdownNetwork(config.tfm_channels, config.topdown.channels,
                             config.topdown.layers, config.topdown.strides,
                             config.topdown.blocktype)
    
    # Build classifier
    if config.bayesian:
        classifier = BayesianClassifier(topdown.out_channels, config.num_class)
    else:
        classifier = LinearClassifier(topdown.out_channels, config.num_class)
    classifier.initialise(config.prior)
    
    # Assemble Pyramid Occupancy Network
    return PyramidOccupancyNetwork(frontend, transformer, topdown, classifier)



def build_variational_encoder_decoder(config):
    
    return VariationalEncoderDecoder(config.num_class, 
                                     config.ved.bottleneck_dim,
                                     config.map_extents,
                                     config.map_resolution)


def build_view_parsing_network(config):

    return VPNModel(1, config.num_class, config.vpn.output_size, 
                    config.vpn.fc_dim, config.map_extents, 
                    config.map_resolution)


