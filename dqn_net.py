
# pylint: disable=F0401

import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()

        hidden_1 = 256
        hidden_2 = 128
        hidden_3 = 64
        # hidden_4 = 64
        # hidden_1 = 2048
        # hidden_2 = 1024
        # hidden_3 = 256

        self.bn1 = nn.BatchNorm1d(hidden_1)
        self.z1 = nn.Linear(state_size, hidden_1)
        self.dropout1 = nn.Dropout(p=0.1)

        self.bn2 = nn.BatchNorm1d(hidden_2)
        self.z2 = nn.Linear(hidden_1, hidden_2)
        self.dropout2 = nn.Dropout(p=0.1)

        self.bn3 = nn.BatchNorm1d(hidden_3)
        self.z3 = nn.Linear(hidden_2, hidden_3)
        self.dropout3 = nn.Dropout(p=0.1)

        self.z4 = nn.Linear(hidden_3, action_size)

        # self.z5 = nn.Linear(hidden_4, action_size)

    def forward(self, h):
        h = F.relu(self.z1(h))
        h = self.dropout1(h)
        # h = self.bn1(h)
        h = F.relu(self.z2(h))
        h = self.dropout2(h)
        # h = self.bn2(h)
        h = F.relu(self.z3(h))
        h = self.dropout3(h)
        # h = self.bn3(h)
        # return F.softmax(self.z4(h))
        # yeah, it doesn't have to be in range of [0,1], it is not really a probability
        return self.z4(h)
