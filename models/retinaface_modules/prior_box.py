
from itertools import product
from math import ceil

import numpy as np
import torch


class PriorBox(object):
    '''
    Anchor generator
    '''

    def __init__(self, cfg, image_size=None, to_tensor=False):
        super(PriorBox, self).__init__()

        self.min_sizes = cfg['min_sizes']
        self.ratios = cfg['ratios']
        self.steps = cfg['steps']
        self.clip = cfg['clip']
        self.image_size = image_size
        self.feature_maps = [[ceil(self.image_size[0]/step),
                              ceil(self.image_size[1]/step)]
                             for step in self.steps]
        self.to_tensor = to_tensor

    def generate(self):
        anchors = []
        for k, f in enumerate(self.feature_maps):
            min_sizes = self.min_sizes[k]
            dense_sizes = [[
                (min_size/self.image_size[0] * ratio[0], min_size/self.image_size[1] * ratio[1])
                for ratio in self.ratios
             ] for min_size in min_sizes]
            dense_ratio = [self.steps[k]/self.image_size[0],
                           self.steps[k]/self.image_size[1]]
            cf = [
                [(c + 0.5) * dense_ratio[0] for c in range(f[0])],
                [(c + 0.5) * dense_ratio[1] for c in range(f[1])],
            ]

            for cy, cx in product(cf[0], cf[1]):
                for dense_size in dense_sizes:
                    for ds_y, ds_x in dense_size:
                        anchors += [cx, cy, ds_x, ds_y]

        anchors = np.array(anchors).reshape(-1, 4)
        if self.clip:
            np.clip(anchors, a_min=0, a_max=1, out=anchors)
        if self.to_tensor:
            anchors = torch.Tensor(anchors)

        return anchors