import math
import numpy as np

class StepLR:
    def __init__(self, optimizer, learning_rate: float, total_epochs: int):
        self.optimizer = optimizer
        self.total_epochs = total_epochs
        self.base = learning_rate
        self.total_steps = total_epochs
        self.min_value = 0
        self.max_value = 0.1
        self.sh = 1
        self.type = 'cos'

    def __call__(self, epoch):
        if self.type =='step':
            if epoch < self.total_epochs * 80/100:
                lr = self.base
            elif epoch < self.total_epochs * 85/100:
                lr = self.base * 0.1
            elif epoch < self.total_epochs * 90/100:
                lr = self.base * 0.1 ** 2
            else:
                lr = self.base * 0.1 ** 3
        elif self.type == 'cos':
            phase = (epoch) / (self.total_steps) * math.pi
            lr = (self.min_value + (self.max_value - self.min_value) * (np.cos(phase) + 1.) / 2.0)# * self.sh
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]
