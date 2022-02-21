# -*- coding: utf-8 -*-
"""
Created on Sun Feb 20 17:10:30 2022

@author: cgg
"""
import torch
import torch.nn as nn
import numpy as np
class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim

        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)

        self.fc = nn.Linear(hidden_dim, output_dim) 

    def forward(self, x):  
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_() 

        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()

        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        out = self.fc(out) 

        return out