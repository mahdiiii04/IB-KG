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
        logvar = torch.clamp(logvar, min=-10, max=10)

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)

        z = mu + eps * std
        return z, mu, logvar
    
class ActionEncoder(nn.Module):
    def __init__(self, latent_dim, model_name, tokenizer_name, device):
        super(ActionEncoder, self).__init__()

        self.device = device
        self.model = BertModel.from_pretrained(model_name)
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
        self.projector = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Linear(512, latent_dim)
        )


    def forward(self, actions):
        inputs = {k: v.to(self.device) for k, v in self.tokenizer(actions, padding=True, truncation=True, return_tensors="pt").items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :]

        projected_embeddings = self.projector(embeddings)

        return projected_embeddings

class CachedActionEncoder(nn.Module):
    def __init__(self, latent_dim, model_name, tokenizer_name, device):
        super(ActionEncoder, self).__init__()

        self.device = device
        self.model = BertModel.from_pretrained(model_name)
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
        self.projector = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Linear(512, latent_dim)
        )
        self.bert_cache = {}  # Cache to store BERT embeddings

    def forward(self, actions):
        # Initialize list to hold BERT embeddings in order of input actions
        bert_embeddings = [None] * len(actions)
        uncached_indices = []
        uncached_actions = []

        # Check cache and collect uncached actions
        for idx, action in enumerate(actions):
            if action in self.bert_cache:
                bert_embeddings[idx] = self.bert_cache[action]
            else:
                uncached_indices.append(idx)
                uncached_actions.append(action)

        # Process uncached actions in a batch
        if uncached_actions:
            # Tokenize with fixed padding and truncation
            inputs = self.tokenizer(
                uncached_actions,
                padding='max_length',
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
            new_embeddings = outputs.last_hidden_state[:, 0, :]

            # Update cache and fill the corresponding positions
            for action, emb in zip(uncached_actions, new_embeddings):
                self.bert_cache[action] = emb.detach().clone()  # Detach and clone to prevent gradient tracking

            for idx, emb in zip(uncached_indices, new_embeddings):
                bert_embeddings[idx] = emb

        # Convert list to tensor and ensure it's on the correct device
        bert_embeddings_tensor = torch.stack(bert_embeddings).to(self.device)

        # Project embeddings using the current projector (allows gradient flow through projector)
        projected_embeddings = self.projector(bert_embeddings_tensor)

        return projected_embeddings
    
class ActionDecoder(nn.Module):
    def __init__(self, latent_dim, model_name, tokenizer_name, device):
        super(ActionDecoder, self).__init__()

        self.device = device

        self.encoder = ActionEncoder(latent_dim, model_name, tokenizer_name, device).to(device)


    def forward(self, valid_actions, zt):

        actions_embeddings = self.encoder.forward(valid_actions)

        eps = 1e-8
        actions_norm = F.normalize(actions_embeddings + eps, p=2, dim=1)
        zt_norm = F.normalize(zt + eps, p=2, dim=0)

        scores = torch.matmul(actions_norm, zt_norm)        

        return scores
    
    def get_action(self, valid_actions, zt, epsilon=0.1):
        scores = self.forward(valid_actions, zt)
        scores = torch.where(torch.isfinite(scores), scores, torch.tensor(-1e8, device=scores.device))

        probs = F.softmax(scores, dim=0)
        log_probs = F.log_softmax(scores, dim=0)

        if torch.rand(1).item() < epsilon:
            action = torch.randint(0, len(valid_actions), (1,)).item()
        else:
            action = torch.multinomial(probs, 1).item()        

        return valid_actions[action], log_probs[action], probs

class Critic(nn.Module):
    def __init__(self, latenet_dim):
        super(Critic, self).__init__()
        self.fc = nn.Linear(latenet_dim, 1)

    def forward(self, x):
        return self.fc(x)

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

        self.action_decoder = ActionDecoder(latent_dim, actor_model_name, actor_tokenizer_name, device).to(device)
        self.critic = Critic(latent_dim).to(device)

    def forward(self, valid_actions, beta=1.0, epsilon=0.1, ib_reg=0.02):
        
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

        l_1 = torch.clamp(l_1, -10.0, 10.0)
        l_2 = torch.clamp(l_2, -10.0, 10.0)

        ib_loss = l_1 - beta * l_2 + ib_reg * l_2.pow(2)

        action, log_prob, probs = self.action_decoder.get_action(valid_actions, z_t, epsilon=epsilon)

        value = self.critic.forward(z_t)

        return ib_loss, action, log_prob, value, probs

    def kl_divergence(self, mu, logvar):
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return kl_div / max(1, mu.size(0))
    

class NOIBKG(nn.Module):
    def __init__(self, max_nodes, feat_dim, rel2id, node2id, hidden_dim, repr_dim, latent_dim, actor_model_name, actor_tokenizer_name, device):
        super(IBKG, self).__init__()
        self.device = device

        self.kg = Grapher(node_mapping=node2id, rel_mapping=rel2id)
        self.node_embedding = nn.Embedding(max_nodes, feat_dim).to(device)
        self.rgcn = RGCN(feat_dim, hidden_dim, repr_dim, num_rels=len(rel2id)).to(device)
        self.attention = AttentionPooling(repr_dim).to(device)

        self.action_decoder = ActionDecoder(repr_dim, actor_model_name, actor_tokenizer_name, device).to(device)
        self.critic = Critic(repr_dim).to(device)

    def forward(self, valid_actions, beta=1.0, epsilon=0.1, ib_reg=0.02):
        node_ids = list(range(len(self.kg.node_mapping)))
        node_ids_tensor = torch.LongTensor(node_ids).to(self.device)
        node_feat = self.node_embedding(node_ids_tensor)

        graph = self.kg.build_graph()
        if hasattr(graph, 'to'):
            graph = graph.to(self.device)
    
        rel_types = torch.tensor(self.kg.get_relations_mapped(), device=self.device)

        h_t = self.rgcn.forward(graph=graph, feat=node_feat, etypes=rel_types)

        graph_repr = self.attention.forward(h_t)

        action, log_prob, probs = self.action_decoder.get_action(valid_actions, graph_repr, epsilon=epsilon)

        # Fixed: Using graph_repr instead of undefined z_t
        value = self.critic.forward(graph_repr)

        return action, log_prob, value, probs

    def kl_divergence(self, mu, logvar):
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return kl_div / max(1, mu.size(0))