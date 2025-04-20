import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import RelGraphConv
from transformers import T5Tokenizer, T5ForConditionalGeneration, BertTokenizer, BertModel
from grapher import Grapher

class TextModel:
    def __init__(self, model_name, tokenizer_name, device, subfolder_name=None):
        self.model_name = model_name
        self.tokenizer_name = tokenizer_name
        self.subfolder_name = subfolder_name

        self.device = device

        self.load_model()

    def load_model(self):
        if self.subfolder_name:
            self.model = T5ForConditionalGeneration.from_pretrained(self.model_name, subfolder=self.subfolder_name).to(self.device)
        else:
            self.model = T5ForConditionalGeneration.from_pretrained(self.model_name).to(self.device)
        self.tokenizer = T5Tokenizer.from_pretrained(self.tokenizer_name)

    def generate_output(self, input_text):
        input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to(self.device)
        output_ids = self.model.generate(input_ids, max_length=50, num_beams=4)
        output_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return output_text


class RGCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_rels):
        super(RGCN, self).__init__()
        self.layer1 = RelGraphConv(in_dim, hidden_dim, num_rels, "basis")
        self.layer2 = RelGraphConv(hidden_dim, out_dim, num_rels, "basis")

    def forward(self, graph, feat, etypes):
        x = self.layer1(graph, feat, etypes)
        x = torch.relu(x)
        x = self.layer2(graph, x, etypes)
        return x
    
class AttentionPooling(nn.Module):
    def __init__(self, input_dim):
        super(AttentionPooling, self).__init__()
        self.attention = nn.Linear(input_dim, 1)

    def forward(self, node_feats):
        scores = self.attention(node_feats)
        att_weights = F.softmax(scores, dim=0)
        pooled = torch.sum(att_weights * node_feats, dim=0)
        return pooled        

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
    
class ActionEncoder(nn.Module):
    def __init__(self, emb_dim, latent_dim, model_name, tokenizer_name):
        super(ActionEncoder, self).__init__()

        self.model = BertModel.from_pretrained(model_name)
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
        self.projector = nn.Linear(emb_dim, latent_dim)


    def forward(self, actions):

        inputs = self.tokenizer(actions, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :]

        projected_embeddings = self.projector(embeddings)

        return projected_embeddings
    
class ActionDecoder(nn.Module):
    def __init__(self, latent_dim, emb_dim, model_name, tokenizer_name):
        super(ActionDecoder, self).__init__()

        self.encoder = ActionEncoder(emb_dim, latent_dim, model_name, tokenizer_name)


    def forward(self, valid_actions, zt):

        actions_embeddings = self.encoder.forward(valid_actions)

        actions_norm = F.normalize(actions_embeddings, p=2, dim=1)
        zt_norm = F.normalize(zt, p=2, dim=0)

        scores = torch.matmul(actions_norm, zt_norm)        

        return scores
    
    def get_action(self, valid_actions, zt, epsilon=0.1):
        scores = self.forward(valid_actions, zt)

        probs = F.softmax(scores, dim=0)
        log_probs = F.log_softmax(scores, dim=0)

        if torch.rand(1).item() < epsilon:
            action = torch.randint(0, len(valid_actions), (1,)).item()
        else:
            action = torch.multinomial(probs, 1).item()        

        return valid_actions[action], log_probs[action]


class IBKG(nn.Module):
    def __init__(self, max_nodes, feat_dim, rel2id, node2id, hidden_dim, repr_dim, latent_dim, actor_model_name, actor_tokenizer_name, device):
        super(IBKG, self).__init__()
        self.device = device

        self.kg = Grapher(node_mapping=node2id, rel_mapping=rel2id)
        self.node_embedding = nn.Embedding(max_nodes, feat_dim).to(device)
        self.rgcn = RGCN(feat_dim, hidden_dim, repr_dim, num_rels=len(rel2id)).to(device)
        self.attention = AttentionPooling(repr_dim).to(device)

        self.ib_encoder = IBEncoder(repr_dim, latent_dim).to(device)
        self.prediction_encoder = IBEncoder(repr_dim, latent_dim).to(device)

        self.action_decoder = ActionDecoder(latent_dim, repr_dim, actor_model_name, actor_tokenizer_name).to(device)

    def forward(self, valid_actions, beta=1.0, epsilon=0.1):
        
        node_ids = list(range(len(self.kg.node_mapping)))
        node_ids_tensor = torch.LongTensor(node_ids).to(self.device)
        node_feat = self.node_embedding(node_ids_tensor)

        graph = self.kg.build_graph()
        if hasattr(graph, 'to'):
            graph = graph.to(self.device)
    
        rel_types = torch.tensor(self.kg.get_relations_mapped(), device=self.device)

        h_t = self.rgcn.forward(graph=graph, feat=node_feat, etypes=rel_types)

        graph_repr = self.attention.forward(h_t)

        z_t, mu_z, logvar_z = self.ib_encoder.forward(graph_repr)

        h_next, mu_h, logvar_h = self.prediction_encoder.forward(graph_repr)

        l_1 = self.kl_divergence(mu_z, logvar_z)
        l_2 = self.kl_divergence(mu_h, logvar_h)

        ib_loss = l_1 - beta * l_2

        action, log_prob = self.action_decoder.get_action(valid_actions, z_t, epsilon=epsilon)

        return ib_loss, action, log_prob

    def kl_divergence(self, mu, logvar):
        return 0.5 * torch.sum(torch.exp(logvar) + mu**2 - 1 - logvar)