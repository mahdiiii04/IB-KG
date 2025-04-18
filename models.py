import torch
import torch.nn as nn
from dgl.nn import RelGraphConv
from transformers import T5Tokenizer, T5ForConditionalGeneration

class TextModel:
    def __init__(self, model_name, tokenizer_name, subfolder_name=None):
        self.model_name = model_name
        self.tokenizer_name = tokenizer_name
        self.subfolder_name = subfolder_name

        self.load_model()

    def load_model(self):
        if self.subfolder_name:
            self.model = T5ForConditionalGeneration.from_pretrained(self.model_name, subfolder=self.subfolder_name)
        else:
            self.model = T5ForConditionalGeneration.from_pretrained(self.model_name)
        self.tokenizer = T5Tokenizer.from_pretrained(self.tokenizer_name)

    def generate_output(self, input_text):
        input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids
        output_ids = self.model.generate(input_ids, max_length=50, num_beams=4)
        output_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return output_text


class RGCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_rels):
        super(RGCN, self).__init__()
        self.layer1 = RelGraphConv(in_dim, hidden_dim, num_rels, "basis")
        self.layer1 = RelGraphConv(hidden_dim, out_dim, num_rels, "basis")

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


