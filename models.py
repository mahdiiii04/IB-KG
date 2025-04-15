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

class IBEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(IBEncoder, self).__init__()

        self.fc_mu = nn.Linear(input_dim, latent_dim)
        self.fc_var = nn.Linear(input_dim, latent_dim)

    def forward(self, x):
        mu = self.fc_mu(x)
        logvar = self.fc_var(x)

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)

        z = mu + eps * std
        return z, mu, logvar


