import torch
import torch.nn as nn

class LossCalculator():
    def __init__(self,
                loss_type,
                weight_class,
                weight_multi_task):
        self.loss_type = loss_type
        self.weight_class = weight_class
        self.weight_multi_task = weight_multi_task

