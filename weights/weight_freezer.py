import torch
import torch.nn as nn

from models.full_model import ContextAwareAttention


class Freezer():
    def __init__(self, model, phase):
        self.model = model
        self.phase = phase
    
    def freeze_weight(self):
        pass