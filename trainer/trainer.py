import numpy as np
import torch
from torchvision.utils import make_grid
from base import BaseTrainer
from utils import inf_loop, MetricTracker


class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self,
                 model,
                 criterion,
                 metric_ftns,
                 optimizer,
                 train_config,
                 val_config,
                 device,
                 data_loader,
                 resume_ckpt_path,
                 valid_data_loader=None,
                 lr_scheduler=None):
        super().__init__(model, criterion, metric_ftns, optimizer, train_config, val_config)

        self.train_config = train_config
        self.val_config = train_config
        self.device = device

        self.train_data_loader = data_loader
        self.len_epoch = len(self.train_data_loader)
        self.valid_data_loader = valid_data_loader

        # Configure each phase and training epochs
        # phase_1: Train Face (Classifier only - Warmup)
        # phase_2: Train Face (All model)
        # phase_3: Train Context (Warmup)
        # phase_4: Train Context (All model + Attention Module)
        # phase_5: Train Gating
        # phase 6: Train all (Face + Context + gating)

        self.start_epoch = self.train_config['start_epoch']
        self.total_epoch = self.train_config['phase_1'] +  \
                           self.train_config['phase_2'] +  \
                           self.train_config['phase_3'] +  \
                           self.train_config['phase_4'] +  \
                           self.train_config['phase_5'] +  \
                           self.train_config['phase_6']

        # self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))

        # Model
        self.model = model
        if resume_ckpt_path is not None:
            self.model.load_state_dict(torch.load(resume_ckpt_path,
                                                    map_location="cpu"))

        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        for batch_idx, (_, _, face_image, context_image, full_image, age_label, emotion_label) \
            in enumerate(self.train_data_loader):

            full_image, context_image, face_image = full_image.to(self.device), context_image.to(self.device), face_image.to(self.device)
            age_label, emotion_label = age_label.to(self.device), emotion_label.to(self.device)

            self.optimizer.zero_grad()
            output_age, output_emotion = self.model(face_image, context_image, full_image)
            outputs = {
                    'age': output_age,
                    'emotion': output_emotion
                }
            labels = {
                    'age': age_label,
                    'emotion': emotion_label
                }
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item())
            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__, met(output, target))

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item()))
                self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

            if batch_idx == self.len_epoch:
                break
        log = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k : v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.valid_data_loader):
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                loss = self.criterion(output, target)

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.valid_metrics.update('loss', loss.item())
                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(output, target))
                self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')
        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
    
    def calculate_loss(self, outputs, labels):
        loss = dict()
        loss['sum'], loss['age'], loss['emotion']= \
            torch.tensor(data=0.).to(self.device), \
            torch.tensor(data=0.).to(self.device), \
            torch.tensor(data=0.).to(self.device)
        for task in outputs.keys():

