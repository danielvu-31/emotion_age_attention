import torch
import torch.nn as nn
import torch.nn.functional as F


class LossCalculator():
    def __init__(self,
                tasks,
                class_stats,
                device,
                use_class_weight=True,
                weight_multi_task="sum"):
        self.tasks = tasks
        self.device = device
        self.use_class_weight = use_class_weight
        self.weight_multi_task = weight_multi_task
        self.criterion = {
            "age": nn.CrossEntropyLoss(),
            "emotion": nn.CrossEntropyLoss()
        }

        if self.use_class_weight:
            self.weight_class = self._calculate_weight_class(class_stats)
            self._weight_per_class()

        if self.weight_multi_task == "uncertainty":
            self.logvar = nn.Parameter(torch.zeros((len(self.tasks)))).to(device)
        else:
            self.logvar = None


    def _weight_per_class(self):
        if self.use_class_weight:
            for t in self.criterion.keys():
                self.criterion[t].weight = self.weight_class[t]

    def _calculate_weight_class(self, class_stats):
        weight_class = {}
        for t in class_stats.keys():
            stat = class_stats[t]
            class_weight = 1/stat
            class_weight /= class_weight.min()
            weight_class[t] = class_weight.to(self.device)
        
        return weight_class
    
    def _weighted_multi_task(self, loss, **kwargs):
        if kwargs["mode"] == "train":
            if self.weight_multi_task == "uncertainty":
                assert self.logvar is not None
                # Init: logvar = nn.Parameter(torch.zeros((task_num)))
                task_weight = torch.exp((-1)*self.logvar)
            elif self.weight_multi_task == "lbtw":
                if 0 in kwargs["initial_loss"].values():
                    task_weight = torch.ones(len(self.tasks))
                else:
                    loss_tasks = torch.Tensor(list(loss.values())).data()
                    loss_ratio = loss_tasks/kwargs["initial_loss"]
                    inverse_traing_rate = loss_ratio
                    task_weight = inverse_traing_rate.pow(kwargs["alpha"])
                    task_weight = task_weight / sum(task_weight) * len(self.tasks)
            else:
                task_weight = torch.Tensor([1. for _ in range(len(self.tasks))])
        else:
            task_weight = torch.Tensor([1. for _ in range(len(self.tasks))])
        return task_weight

    def _compute_loss_per_task(self, outputs, labels):
        loss = dict()
        loss['sum'], loss['age'], loss['emotion']= \
            torch.tensor(data=0.).to(self.device), \
            torch.tensor(data=0.).to(self.device), \
            torch.tensor(data=0.).to(self.device)
        for _, t in enumerate(self.tasks):
            loss[t] = self.criterion[t](outputs[t], labels[t])
        return loss

    def _compute_weighted_sum(self, loss, weight_task):
        for idx, t in enumerate(self.tasks):
            weighted_loss = torch.mul(weight_task[idx].to(self.device),
                                    loss[t])
            loss['sum'] += weighted_loss

        if self.weight_multi_task == "uncertainty":
            loss['sum'] += torch.log(1+torch.exp(self.logvar))

        return loss

