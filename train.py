
import os
from datetime import datetime
from argparse import ArgumentParser
from progressbar import progressbar

import torch
import torch.nn as nn
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter

from src.models.model_factory import build_model, build_criterion
from src.data.data_factory import build_dataloaders
from src.utils.configs import get_default_configuration, load_config
from src.utils.confusion import BinaryConfusionMatrix
from src.data.nuscenes.utils import NUSCENES_CLASS_NAMES
from src.data.argoverse.utils import ARGOVERSE_CLASS_NAMES

def train(dataloader, model, criterion, optimiser, summary, config, epoch):

    model.train()

    # Initialise confusion matrix
    confusion = BinaryConfusionMatrix(config.num_class)
    
    # Iterate over dataloader
    iteration = (epoch - 1) * len(dataloader)
    for i, batch in enumerate(progressbar(dataloader, poll_interval=1)):

        # Move tensors to GPU
        if len(config.gpus) > 0:
            batch = [t.cuda() for t in batch]
        
        # Predict class occupancy scores and compute loss
        image, calib, labels, mask = batch
        if config.model == 'ved':
            logits, mu, logvar = model(image)
            loss = criterion(logits, labels, mask, mu, logvar)
        else:
            logits = model(image, calib)
            loss = criterion(logits, labels, mask)


        # Compute gradients and update parameters
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        # Update confusion matrix
        scores = logits.sigmoid()
        confusion.update(scores > config.score_thresh, labels, mask)

        # Update tensorboard
        if i % config.log_interval == 0:
            summary.add_scalar('train/loss', float(loss), iteration)

        iteration += 1

    # Print and record results
    display_results(confusion, config.dataset)
    log_results(confusion, config.dataset, summary, 'train', epoch)



def evaluate(dataloader, model, criterion, summary, config, epoch):

    model.eval()

    # Initialise confusion matrix
    confusion = BinaryConfusionMatrix(config.num_class)
    
    # Iterate over dataset
    for i, batch in enumerate(progressbar(dataloader, poll_interval=1)):

        # Move tensors to GPU
        if len(config.gpus) > 0:
            batch = [t.cuda() for t in batch]
        
        # Predict class occupancy scores and compute loss
        image, calib, labels, mask = batch
        if config.model == 'ved':
            logits, mu, logvar = model(image)
            loss = criterion(logits, labels, mask, mu, logvar)
        else:
            logits = model(image, calib)
            loss = criterion(logits, labels, mask)

        # Update confusion matrix
        scores = logits.sigmoid()
        confusion.update(scores > config.score_thresh, labels, mask)

        # Update tensorboard
        if i % config.log_interval == 0:
            summary.add_scalar('val/loss', float(loss), iteration)

        iteration += 1

    # Print and record results
    display_results(confusion, config.dataset)
    log_results(confusion, config.dataset, summary, 'val', epoch)

    return confusion.mean_iou



def display_results(confusion, dataset):

    # Display confusion matrix summary
    class_names = NUSCENES_CLASS_NAMES if dataset == 'nuscenes' \
        else ARGOVERSE_CLASS_NAMES
    
    print('\nResults:')
    for name, iou_score in zip(class_names, confusion.iou):
        print('{:20s} {:.3f}'.format(name, iou_score)) 
    print('{:20s} {:.3f}'.format('MEAN', confusion.mean_iou))



def log_results(confusion, dataset, summary, split, epoch):

    # Display and record epoch IoU scores
    class_names = NUSCENES_CLASS_NAMES if dataset == 'nuscenes' \
        else ARGOVERSE_CLASS_NAMES

    for name, iou_score in zip(class_names, confusion.iou):
        summary.add_scalar(f'{split}/iou/{name}', iou_score, epoch)
    summary.add_scalar(f'{split}/iou/MEAN', confusion.mean_iou, epoch)



def save_checkpoint(path, model, optimizer, scheduler, epoch, best_iou):

    if isinstance(model, nn.DataParallel):
        model = model.module
    
    ckpt = {
        'model' : model.state_dict(),
        'optimizer' : optimizer.state_dict(),
        'scheduler' : scheduler.state_dict(),
        'epoch' : epoch,
        'best_iou' : best_iou
    }

    torch.save(ckpt, path)


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

    # Restore config from an existing experiment
    if args.resume is not None:
        config.merge_from_file(os.path.join(args.resume, 'config.yml'))
    
    # Override with command line options
    config.merge_from_list(args.options)

    # Finalise config
    config.freeze()

    return config


def create_experiment(config, tag, resume=None):

    # Restore an existing experiment if a directory is specified
    if resume is not None:
        print("\n==> Restoring experiment from directory:\n" + resume)
        logdir = resume
    else:
        # Otherwise, generate a run directory based on the current time
        name = datetime.now().strftime('{}_%y-%m-%d--%H-%M-%S').format(tag)
        logdir = os.path.join(os.path.expandvars(config.logdir), name)
        print("\n==> Creating new experiment in directory:\n" + logdir)
        os.makedirs(logdir)
    
    # Display the config options on-screen
    print(config.dump())
    
    # Save the current config
    with open(os.path.join(logdir, 'config.yml'), 'w') as f:
        f.write(config.dump())
    
    return logdir



    




def main():

    parser = ArgumentParser()
    parser.add_argument('--tag', type=str, default='run',
                        help='optional tag to identify the run')
    parser.add_argument('--dataset', choices=['nuscenes', 'argoverse'],
                        default='nuscenes', help='dataset to train on')
    parser.add_argument('--model', choices=['pyramid', 'vpn', 'ved'],
                        default='pyramid', help='model to train')
    parser.add_argument('--resume', default=None, 
                        help='path to an experiment to resume')
    parser.add_argument('--options', nargs='*', default=[],
                        help='list of addition config options as key-val pairs')
    args = parser.parse_args()

    # Load configuration
    config = get_configuration(args)
    
    # Create a directory for the experiment
    logdir = create_experiment(config, args.tag, args.resume)

    # Create tensorboard summary 
    summary = SummaryWriter(logdir)

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
    if args.resume:
        epoch, best_iou = load_checkpoint(os.path.join(logdir, 'latest.pth'),
                                          model, optimiser, lr_scheduler)
    else:
        epoch, best_iou = 1, 0

    # Main training loop
    while epoch <= config.num_epochs:

        # Train model for one epoch
        train(train_loader, model, criterion, optimiser, summary, config, epoch)

        # Evaluate on the validation set
        val_iou = evaluate(val_loader, model, criterion, summary, config, epoch)

        # Update learning rate
        lr_scheduler.step()

        # Save checkpoints
        if val_iou > best_iou:
            best_iou = val_iou
            save_checkpoint(os.path.join(logdir, 'best.pth'), model, 
                            optimiser, lr_scheduler, epoch, best_iou)
        
        save_checkpoint(os.path.join(logdir, 'latest.pth'), model, optimiser, 
                        lr_scheduler, epoch, best_iou)
    
    print("\nTraining complete!")



if __name__ == '__main__':
    main()

                



    











