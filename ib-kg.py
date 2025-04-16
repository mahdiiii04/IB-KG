import torch
import torch.nn as nn
from models import *
from grapher import Grapher


class IBKG(nn.Module):
    def __init__(self, max_nodes, feat_dim, rel2id, hidden_dim, repr_dim, latent_dim):
        super(IBKG, self).__init__()

        self.kg = Grapher(node_mapping={}, rel_mapping=rel2id)
        self.node_embedding = nn.Embedding(max_nodes, feat_dim)
        self.rgcn = RGCN(feat_dim, hidden_dim, repr_dim, num_rels=len(rel2id))

        self.ib_encoder = IBEncoder(repr_dim, latent_dim)
        self.prediction_encoder = IBEncoder(repr_dim, latent_dim)

    def forward(self, beta=1.0):
        
        node_ids = self.kg.get_nodes_mapped()
        node_feat = self.node_embedding(node_ids)

        graph = self.kg.build_graph()
        rel_types = self.kg.get_relations_mapped()

        h_t = self.rgcn.forward(graph=graph, feat=node_feat, etypes=rel_types)

        z_t, mu_z, logvar_z = self.ib_encoder.forward(h_t)

        h_next, mu_h, logvar_h = self.prediction_encoder.forward(h_t)

        l_1 = self.kl_divergence(mu_z, logvar_z)
        l_2 = self.kl_divergence(mu_h, logvar_h)

        loss = l_1 - beta * l_2

        return loss, z_t
        
    def kl_divergence(self, mu, logvar):
        return 0.5 * torch.sum(torch.exp(logvar) + mu**2 - 1 - logvar)