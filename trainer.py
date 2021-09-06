import numpy as np
from tqdm import tqdm
import torch
from base import BaseTrainer
from weights import Freezer
from loss import LossCalculator
from utils import log_tensorboard 


class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self,
                 model,
                 tasks,
                 criterion,
                 optimizer,
                 config,
                 device,
                 train_loader,
                 writer,
                 resume_ckpt_path=None,
                 val_loader=None,
                 lr_scheduler=None):
        super().__init__(tasks, model, criterion, optimizer, device, config)

        self.train_loader = train_loader
        self.len_epoch = len(self.train_data_loader)
        self.val_loader = val_loader
        self.writer = writer

        self.lr_scheduler = lr_scheduler
        self.lr_count_step = 0

        # Model
        self.model = model
        if resume_ckpt_path is not None:
            self.model.load_state_dict(torch.load(resume_ckpt_path,
                                                    map_location="cpu"))


    def _train_epoch(self, epoch):
        # Initialize a dictionary to store values
        train_stats = {
            'train_loss': {
                'emotion': 0,
                'age': 0,
                'sum': 0
            }
        }

        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        initial_loss_list = torch.zeros(len(self.tasks)).to(self.device)

        train_pbar = tqdm(self.train_loader, desc=f'[EPOCH] {epoch + 1}')

        for _, (_, _, face_image, context_image, full_image, age_label, emotion_label) \
            in enumerate(train_pbar):

            full_image, context_image, face_image = full_image.to(self.device), context_image.to(self.device), face_image.to(self.device)
            age_label, emotion_label = age_label.to(self.device), emotion_label.to(self.device)

            output_age, output_emotion = self.model(face_image, context_image, full_image)

            outputs = {
                    'age': output_age,
                    'emotion': output_emotion
                }
            labels = {
                    'age': age_label,
                    'emotion': emotion_label
                }

            loss = self.criterion._compute_loss_per_task(outputs, labels)
            task_weights = self.criterion._weighted_multi_task(loss,
                                                               initial_loss=initial_loss_list,
                                                               alpha=0.5)
            loss = self.criterion._compute_weighted_sum(loss, task_weights)
            
            if self.config['optimizer']['type'] == "sam":
                loss['sum'].backward()
                self.optimizer.first_step(zero_grad=True)
                # second forward-backward pass
                output_age, output_emotion = self.model(face_image, context_image, full_image)
                
                outputs = {
                    'age': output_age,
                    'emotion': output_emotion
                }

                labels = {
                    'age': age_label,
                    'emotion': emotion_label
                }
                
                loss = self.criterion._compute_loss_per_task(outputs, labels)
                task_weights = self.criterion._weighted_multi_task(loss,
                                                                initial_loss=initial_loss_list,
                                                                alpha=0.5,
                                                                mode="train")
                loss = self.criterion._compute_loss_per_task(loss, task_weights)
                loss['sum'].backward()
                self.optimizer.second_step(zero_grad=True)
            elif self.config['optimizer']['type'] == "pcgrad":
                losses = [loss[t] for t in self.tasks]
                self.optimizer.zero_grad()
                self.optimizer.pc_backward(losses)
                self.optimizer.step()
            else:
                self.optimizer.zero_grad()
                loss['sum'].backward()
                self.optimizer.step()

            # Result tqdm
            for k in train_stats['train_loss'].keys():
                self.logger['train_loss'][k] += loss[k].detach().item()
                train_pbar.set_postfix({
                    'loss_age': loss['age'].item(),
                    'loss_emotion': loss['emotion'].item(),
                    'loss_sum': loss['sum'].item()
                })

        # Validation
        val_stats = self._valid_epoch(epoch)
        log_tensorboard(self.writer, self.tasks, self.phase, train_stats, "train", epoch)
        log_tensorboard(self.writer, self.tasks, self.phase, val_stats, "val", epoch)

        return train_stats, val_stats

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()

        val_stats = {
            'val_loss': {
                'emotion': 0,
                'age': 0,
                'sum': 0
            }, 
            'val_accuracy': {
                'emotion': 0,
                'age': 0,
            }
        }

        corrects = {
            'age': 0,
            'total_age': 0,
            'emotion': 0,
            'total_emotion': 0,
        }

        val_pbar = tqdm(self.val_loader, desc=f'[VALIDATION] {epoch + 1}')
        with torch.no_grad():
            for _, (_, _, face_image, context_image, full_image, age_label, emotion_label) in enumerate(val_pbar):
    
                full_image, context_image, face_image = full_image.to(self.device), context_image.to(self.device), face_image.to(self.device)
                age_label, emotion_label = age_label.to(self.device), emotion_label.to(self.device)

                output_age, output_emotion = self.model(face_image, context_image, full_image)
                

                outputs = {
                        'age': output_age,
                        'emotion': output_emotion
                    }
                labels = {
                        'age': age_label,
                        'emotion': emotion_label
                    }

                val_loss = self.criterion._compute_loss_per_task(outputs, labels)

                task_weights = self.criterion._weighted_multi_task(val_loss,
                                                                    initial_loss=None,
                                                                    alpha=0.5,
                                                                    mode="val")
                val_loss = self.criterion._compute_weighted_sum(val_loss, task_weights)

                val_pbar.set_postfix({
                    'loss_emotion': val_loss['emotion'].item(),
                    'loss_age': val_loss['age'].item(),
                    'loss_sum': val_loss['sum'].item()
                })

                for t in self.tasks:
                    self.val_stats['val_loss'][t] += val_loss[t].item()*len(labels[t])

                if outputs['age'].shape[1] == self.num_emotion_classes:
                    _, out_age_pred = torch.max(outputs['age'], 1)
                    age_corrects = torch.sum(out_age_pred == labels['age'])
                    corrects['age'] += age_corrects.item()
                    corrects['total_age'] += len(labels['age'])

                if outputs['emotion'].shape[1] == self.num_emotion_classes:
                    _, out_emotion_pred = torch.max(outputs['emotion'], 1)
                    emo_corrects = torch.sum(out_emotion_pred == labels['emotion'])
                    corrects['emotion'] += emo_corrects.item()
                    corrects['total_emotion'] += len(labels['emotion'])
                
            # Accuracy & Loss Summary
            for t in self.tasks:
                # Loss
                self.val_stats['val_loss'][t] /= corrects[f'total_{t}']
                self.val_stats['val_loss']["sum"] += self.val_stats['val_loss'][t]

                print(f"---Val Loss for {t.upper()}: {self.val_stats['val_loss'][t]}")

                # Accuracy
                self.val_stats['val_accuracy'][t] = float(corrects[t] / corrects[f'total_{t}'])
                print(f"---Val Acc for {t.upper()}: {self.val_stats['val_accuracy'][t]}")
            
            print(f"---Val Total Loss: {self.val_stats['val_loss']['sum']}")
        
            # Step LR
            if self.lr_scheduler is not None and \
                self.val_stats['val_loss']['sum'] > self.best_val_loss:
                self.lr_count_step += 1
                if self.lr_count_step == self.cfg_trainer['lr_threshold']:
                    print('Stepping LR...')
                    self.lr_scheduler.step()
                    self.lr_count_step = 0

        return val_stats
