import dgl
import torch
import torch.nn as nn
from dgl.nn import RelGraphConv

class RGCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_rels):
        super(RGCN, self).init()
        self.layer1 = RelGraphConv(in_dim, hidden_dim, num_rels, "sum")
        self.layer1 = RelGraphConv(hidden_dim, out_dim, num_rels, "sum")

    def forward(self, graph, feat, etypes):
        x = self.layer1(graph, feat, etypes)
        x = torch.relu(x)
        x = self.layer2(graph, x, etypes)
        return x
    
