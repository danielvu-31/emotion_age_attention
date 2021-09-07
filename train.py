import argparse
import collections
import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import data_loader.data_loader as module_data
from loss import LossCalculator
from optimizer import load_optimizer
from parse_config import ConfigParser
from trainer import Trainer
from models import ContextAwareAttention


# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


def main(config):
    logger = config.get_logger('train')

    # setup data_loader instances
    train_loader = config.init_obj('train_loader', module_data)
    val_loader = config.init_obj('val_loader', module_data)

    # build model architecture, then print to console
    model = config.init_obj('arch', ContextAwareAttention)
    logger.info(model)
    
    device = f"cuda:{config['gpu_id']}"
    model = model.to(device)

    # get function handles of loss and metrics
    criterion = LossCalculator(config['gpu_id'],
                               train_loader.class_statistics,
                               device,
                               config['loss']['weight_class'],
                               config['loss']['multi_task_type'])

    optimizer = load_optimizer(model, 0.001, config['optimizer']['type'])
    lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)
    
    # Writer
    writer = SummaryWriter(log_dir="logs/")
    trainer = Trainer(
        model=model,
        tasks=['age', 'emotion'],
        criterion=criterion,
        optimizer=optimizer,
        config=config,
        device=device,
        train_loader=train_loader,
        val_loader=val_loader,
        writer=writer,
        resume_ckpt_path=config['trainer']['resume_ckpt_path'],
        lr_scheduler=lr_scheduler
        )
    
    trainer.train()


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size')
    ]
    config = ConfigParser.from_args(args, options)
    main(config)
