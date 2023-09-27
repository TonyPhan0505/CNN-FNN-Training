import torch
import torch.nn as nn
import torch.nn.functional as F

class FNN(nn.Module):
    def __init__(self, loss_type, num_classes):
        super(FNN, self).__init__()
        self.loss_type = loss_type
        self.num_classes = num_classes
        self.model = nn.Sequential(
            nn.Linear(784, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, self.num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        output = self.model(x)
        return output

    def get_loss(self, output, target):
        loss = None
        if self.loss_type == "l2":
            loss = F.mse_loss(output, target.float().view_as(output))
        elif self.loss_type == "ce":
            loss = F.cross_entropy(output, target)
        else:
            raise AssertionError(f'invalid loss type: {self.loss_type}')
        return loss