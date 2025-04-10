from models import *
import torch
import torch.nn as nn



class IBKG(nn.Module):
    def __init__(self):
        super(IBKG, self).__init__()
        self.RGCN = RGCN(in_dim=100, hidden_dim=200, out_dim=100, num_rels=len(rel2id))

    def forward(self):
        pass