import torch
import torch.nn as nn
from models import *
from grapher import Grapher


class IBKG(nn.Module):
    def __init__(self, max_nodes, feat_dim, rel2id, hidden_dim, out_dim, latent_dim):
        super(IBKG, self).__init__()

        self.kg = Grapher(node_mapping={}, rel_mapping=rel2id)
        self.node_embedding = nn.Embedding(max_nodes, feat_dim)
        self.rgcn = RGCN(feat_dim, hidden_dim, out_dim, num_rels=len(rel2id))

        self.ib_encoder = IBEncoder(out_dim, )

    def forward(self):
        pass