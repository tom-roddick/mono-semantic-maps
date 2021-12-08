import os
from datetime import datetime
from argparse import ArgumentParser
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR

from src.models.model_factory import build_model, build_criterion
from src.data.data_factory import build_dataloaders
from src.utils.configs import get_default_configuration, load_config
from src.utils.confusion import BinaryConfusionMatrix
from src.data.nuscenes.utils import NUSCENES_CLASS_NAMES
from src.data.argoverse.utils import ARGOVERSE_CLASS_NAMES
from src.utils.visualise import colorise

from PIL import Image
import numpy as np

def infer(dataloader, model, criterion, config, vis_dir):

    model.eval()

    # Compute prior probability of occupancy
    prior = torch.tensor(config.prior)
    prior_log_odds = torch.log(prior / (1 - prior))

    # Initialise confusion matrix
    confusion = BinaryConfusionMatrix(config.num_class)
    
    counter = 0
    # Iterate over dataset
    for i, batch in enumerate(tqdm(dataloader)):

        # Move tensors to GPU
        if len(config.gpus) > 0:
            batch = [t.cuda() for t in batch]
        
        # Predict class occupancy scores and compute loss
        image, calib, labels, mask = batch
        with torch.no_grad():
            if config.model == 'ved':
                logits, mu, logvar = model(image)
                loss = criterion(logits, labels, mask, mu, logvar)
            else:
                logits = model(image, calib)
                loss = criterion(logits, labels, mask)

        # Update confusion matrix
        scores = logits.cpu().sigmoid()  
        confusion.update(scores > config.score_thresh, labels, mask)

        batch_size = len(image) # should be 1 though..
        counter += batch_size
        for i in range(batch_size):
            img_idx = counter + i
            save_visualization(image[i], scores[i] > config.score_thresh, labels[i], mask[i], img_idx, config.train_dataset, vis_dir)


    # Print and record results
    display_results(confusion, config.train_dataset)

    return confusion.mean_iou


def save_visualization(image, scores, labels, mask, step, dataset, save_dir):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    class_names = NUSCENES_CLASS_NAMES if dataset == 'nuscenes' \
        else ARGOVERSE_CLASS_NAMES

    mask = mask.detach().cpu()
    scores = scores * mask
    labels = labels.detach().cpu() * mask
    gt_map = colorise(labels, 'coolwarm', 0, 1) * 255
    bev_map = colorise(scores, 'coolwarm', 0, 1) * 255

    gt_map = gt_map.astype(np.uint8)
    bev_map = bev_map.astype(np.uint8)

    obstacles = mask == 0
    gt_map[:, obstacles] = 0
    bev_map[:, obstacles] = 0

    # gt_map = gt_map * mask.detach().cpu().numpy()
    # bev_map = bev_map * mask.detach().cpu().numpy()

    
    for i in range(3): # only visualize first three classes
        cls_name = class_names[i]
        gt_dir = os.path.join(save_dir, cls_name, 'gt')
        pred_dir = os.path.join(save_dir, cls_name, 'pred')

        if not os.path.exists(gt_dir):
            os.makedirs(gt_dir)
        if not os.path.exists(pred_dir):
            os.makedirs(pred_dir)

        Image.fromarray(gt_map[i]).save(os.path.join(gt_dir, '{}.jpg'.format(step)))
        Image.fromarray(bev_map[i]).save(os.path.join(pred_dir, '{}.jpg'.format(step)))

    img_dir = os.path.join(save_dir, 'images')
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    image = image.cpu().numpy() * 255
    image = image.astype(np.uint8)
    image = np.moveaxis(image, 0, -1)
    Image.fromarray(image).save(os.path.join(img_dir, '{}.jpg'.format(step)))


def display_results(confusion, dataset):

    # Display confusion matrix summary
    class_names = NUSCENES_CLASS_NAMES if dataset == 'nuscenes' \
        else ARGOVERSE_CLASS_NAMES
    
    print('\nResults:')
    for name, iou_score in zip(class_names, confusion.iou):
        print('{:20s} {:.3f}'.format(name, iou_score)) 
    print('{:20s} {:.3f}'.format('MEAN', confusion.mean_iou))



def load_checkpoint(path, model, optimizer, scheduler):
    
    ckpt = torch.load(path)

    # Load model weights
    if isinstance(model, nn.DataParallel):
        model = model.module
    model.load_state_dict(ckpt['model'])

    # Load optimiser state
    optimizer.load_state_dict(ckpt['optimizer'])

    # Load scheduler state
    scheduler.load_state_dict(ckpt['scheduler'])

    return ckpt['epoch'], ckpt['best_iou']



# Load the configuration for this experiment
def get_configuration(args):

    # Load config defaults
    config = get_default_configuration()

    # Load dataset options
    config.merge_from_file(f'configs/datasets/{args.dataset}.yml')

    # Load model options
    config.merge_from_file(f'configs/models/{args.model}.yml')

    # Load experiment options
    config.merge_from_file(f'configs/experiments/{args.experiment}.yml')

    # Restore config from an existing experiment
    if args.ckpt_dir is not None:
        config.merge_from_file(os.path.join(args.ckpt_dir, 'config.yml'))
    
    # Override with command line options
    config.merge_from_list(args.options)

    # set batch size to 1 for infer
    config.batch_size = 1

    # Finalise config
    config.freeze()

    return config



def main():

    parser = ArgumentParser()
    parser.add_argument('--tag', type=str, default='run',
                        help='optional tag to identify the run')
    parser.add_argument('--dataset', choices=['nuscenes', 'argoverse'],
                        default='nuscenes', help='dataset to train on')
    parser.add_argument('--model', choices=['pyramid', 'vpn', 'ved'],
                        default='pyramid', help='model to train')
    parser.add_argument('--experiment', default='test', 
                        help='name of experiment config to load')
    parser.add_argument('--ckpt_dir', default=None, 
                        help='path to checkpoint dir')
    parser.add_argument('--vis_dir', default=None, 
                        help='folder to save visualization')
    parser.add_argument('--options', nargs='*', default=[],
                        help='list of addition config options as key-val pairs')
    args = parser.parse_args()

    # Load configuration
    config = get_configuration(args)
    
    # Set default device
    if len(config.gpus) > 0:
        torch.cuda.set_device(config.gpus[0])

    # Setup experiment
    model = build_model(config.model, config)
    criterion = build_criterion(config.model, config)
    train_loader, val_loader = build_dataloaders(config.train_dataset, config)

    # Build optimiser and learning rate scheduler
    optimiser = SGD(model.parameters(), config.learning_rate, 
                    weight_decay=config.weight_decay)
    lr_scheduler = MultiStepLR(optimiser, config.lr_milestones, 0.1)

    # Load checkpoint
    if args.ckpt_dir:
        epoch, best_iou = load_checkpoint(os.path.join(args.ckpt_dir, 'best.pth'),
                                          model, optimiser, lr_scheduler)
    else:
        print('No checkpoint provided!')

    # Evaluate on the validation set
    val_iou = infer(val_loader, model, criterion, config, args.vis_dir)

    print("\nInfer complete!")



if __name__ == '__main__':
    main()