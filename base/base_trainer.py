import os
import torch
from abc import abstractmethod
from numpy import inf
from logger import TensorboardWriter
from weights import Freezer
from data_loader import FaceDataset
from optimizer import load_optimizer


class BaseTrainer:
    """
    Base class for all trainers
    """
    def __init__(self, config, tasks, model, criterion, optimizer, weight_control, device):
        self.config = config
        self.logger = config.get_logger('trainer', config['trainer']['verbosity'])

        self.model = model
        self.tasks = tasks
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.weight_control = Freezer(self.model)

        cfg_trainer = config['trainer']
        self.start_epoch = cfg_trainer['start_epoch']

        # Configure each phase and training epochs
        # phase_1: Train Face (Classifier only - Warmup)
        # phase_2: Train Face (All model)
        # phase_3: Train Context (Warmup)
        # phase_4: Train Context (All model + Attention Module)
        # phase_5: Train Gating
        # phase 6: Train all (Face + Context + gating)

        self.epochs = cfg_trainer['phase_1'] +  \
                        cfg_trainer['phase_2'] +  \
                        cfg_trainer['phase_3'] +  \
                        cfg_trainer['phase_4'] +  \
                        cfg_trainer['phase_5'] +  \
                        cfg_trainer['phase_6']

        self.improved_loss, self.improved_age, self.improved_emotion = False, False, False
        self.save_period = cfg_trainer['save_period']
        self.monitor = cfg_trainer.get('monitor', 'off')

        # configuration to monitor model performance and save best
        if self.monitor == 'off':
            self.mnt_mode = 'off'
            self.mnt_best = 0
        else:
            self.mnt_mode, self.mnt_metric = self.monitor.split() # mnt_metric should be val_accuracy/val_loss
            assert self.mnt_mode in ['min', 'max']

            self.mnt_best = {
                "val_loss": inf,
                "val_accuracy": {
                    "age": -inf,
                    "emotion": -inf
                }
            }

            self.early_stop = cfg_trainer.get('early_stop', inf)
            if self.early_stop <= 0:
                self.early_stop = inf

        self.start_epoch = 1
        self.checkpoint_dir = config.save_dir
        self.phase = 1

        # TODO: setup visualization writer instance                
        self.writer = TensorboardWriter(config.log_dir, self.logger, cfg_trainer['tensorboard'])

        if config.resume is not None:
            self._resume_checkpoint(config.resume)

    @abstractmethod
    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current epoch number
        """
        raise NotImplementedError

    def train(self):
        """
        Full training logic
        """
        not_improved_count = 0
        for epoch in range(self.start_epoch, self.epochs + 1):
            phase, lr = self._check_phase(epoch)
            if phase != self.phase:
                self.phase = phase
                self.weight_control.freeze_weight(phase)
                self.model = self.weight_control._return_model()
                self.init_train_cfg(lr)

                # Re-init eval metric
                self.mnt_best = {
                            "val_loss": inf,
                            "val_accuracy": {
                                "age": -inf,
                                "emotion": -inf
                            }
                        }

            result_train, result_val = self._train_epoch(epoch)

            # save logged informations into log dict
            log = {'epoch': epoch}
            log.update(result_train)
            log.update(result_val)

            # print logged informations to the screen
            for key, value in log.items():
                if type(value) != dict:
                    self.logger.info('    {:15s}: {}'.format(str(key), value))
                else:
                    self.logger.info('    {:15s}: '.format(str(key)))
                    for k, v in value.items():
                        self.logger.info('    {:15s}: {}'.format(k, v))

            # evaluate model performance according to configured metric, save best checkpoint as model_best
            best = False
            if self.mnt_mode != 'off':
                try:
                    # check whether model performance improved or not, according to specified metric(mnt_metric)
                    # We assume the objective function is MIN LOSS or MAX ACCURACY
                    self.improved_loss = self.mnt_mode == 'min' and log[self.mnt_metric]["sum"] <= self.mnt_best
                    self.improved_age =  self.mnt_mode == 'max' and log[self.mnt_metric]["age"] >= self.mnt_best[self.mnt_metric]["age"]
                    self.improved_emotion = self.mnt_mode == 'max' and log[self.mnt_metric]["emotion"] >= self.mnt_best[self.mnt_metric]["emotion"]
                    improved = self.improved_loss or self.improved_age or self.improved_emotion
                except KeyError:
                    self.logger.warning("Warning: Metric '{}' is not found. "
                                        "Model performance monitoring is disabled.".format(self.mnt_metric))
                    self.mnt_mode = 'off'
                    improved = False

                if improved:
                    if self.mnt_mode == 'min':
                        self.mnt_best[self.mnt_metric] = log[self.mnt_metric]["sum"]
                    else:
                        if self.improved_age:
                            self.mnt_best[self.mnt_metric]["age"] = log[self.mnt_metric]["age"]
                        if self.improved_emotion:
                            self.mnt_best[self.mnt_metric]["emotion"] = log[self.mnt_metric]["emotion"]
                    not_improved_count = 0
                    best = True
                else:
                    not_improved_count += 1

                if not_improved_count > self.early_stop:
                    self.logger.info("Validation performance didn\'t improve for {} epochs. "
                                     "Training stops.".format(self.early_stop))
                    break

            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch, save_best=best)

    def _check_phase(self, epoch):
        phase_list = []
        phase_threshold = 0
        for index in range(6):
            phase_threshold += self.config['trainer'][f'phase_{index+1}']
            phase_list[index] = phase_threshold

        if epoch <= phase_list[0]:
            phase = 1
            lr = self.config['lr_phase']['phase1']
        elif phase_list[1] < epoch <= phase_list[2]:
            phase = 2
            lr = self.config['lr_phase']['phase2']
        elif phase_list[2] < epoch <= phase_list[3]:
            phase = 3
            lr = self.config['lr_phase']['phase3']
        elif phase_list[3] < epoch <= phase_list[4]:
            phase = 4
            lr = self.config['lr_phase']['phase4']
        elif phase_list[4] < epoch <= phase_list[5]:
            phase = 5
            lr = self.config['lr_phase']['phase5']
        else:
            phase = 6
            lr = self.train_configs['lr_all']

            return phase, lr

    def _save_checkpoint(self,
                        epoch):
        """
        Saving checkpoints

        :param epoch: current epoch number
        :param log: logging information of the epoch
        :param save_best: if True, rename the saved checkpoint to 'best-().pth'
        """
        arch = type(self.model).__name__
        state = {
            'arch': arch,
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.mnt_best,
            'config': self.config
        }

        if self.improved_loss:
            filename = os.path.join(self.checkpoint_dir / 'best-loss.pth')
            self.logger.info("Saving checkpoint new best checkpoint based on val loss: {} ...".format(filename))
            torch.save(state, filename)
        if self.improved_age:
            filename = os.path.join(self.checkpoint_dir / 'best-age.pth')
            self.logger.info("Saving checkpoint new best checkpoint based on age accuracy: {} ...".format(filename))
            torch.save(state, filename)
        if self.improved_emotion:
            filename = os.path.join(self.checkpoint_dir / 'best-emotion.pth')
            self.logger.info("Saving checkpoint new best checkpoint based on emotion accuracy: {} ...".format(filename))
            torch.save(state, filename)

        filename = str(self.checkpoint_dir / 'checkpoint-epoch{}.pth'.format(epoch))
        torch.save(state, filename)
        self.logger.info("Saving checkpoint: {} ...".format(filename))

    def _resume_checkpoint(self, resume_path):
        # TODO: Review this function again
        """
        Resume from saved checkpoints

        :param resume_path: Checkpoint path to be resumed
        """
        resume_path = str(resume_path)
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.mnt_best = checkpoint['monitor_best']

        # load architecture params from checkpoint.
        if checkpoint['config']['arch'] != self.config['arch']:
            self.logger.warning("Warning: Architecture configuration given in config file is different from that of "
                                "checkpoint. This may yield an exception while state_dict is being loaded.")
        self.model.load_state_dict(checkpoint['state_dict'])

        # load optimizer state from checkpoint only when optimizer type is not changed.
        if checkpoint['config']['optimizer']['type'] != self.config['optimizer']['type']:
            self.logger.warning("Warning: Optimizer type given in config file is different from that of checkpoint. "
                                "Optimizer parameters not being resumed.")
        else:
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        self.logger.info("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))
    

    def init_train_cfg(self, lr):
        # Dataloaders
        self.train_loader = self.config.init_obj('train_loader', FaceDataset)
        self.val_loader = self.config.init_obj('val_loader', FaceDataset)

        # Optimizer
        self.optimizer = load_optimizer(self.model, lr, self.config['optimizer']['type'])

        # Best last phase checkpoint
        self.model.load_state_dict(torch.load(os.path.join(self.checkpoint_dir / 'best-loss.pth'),
                                                map_location="cpu"))

