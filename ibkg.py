import torch
import torch.nn as nn
import torch.optim as optim
import yaml
import json
from models import *
from OBSRVR import OBSRVR
from injection import ConceptGraph
from env import TextEnv
from logger import Logger

with open('conf/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

with open('mapping/rel2id.json', 'r') as f:
    rel2id = json.load(f)

obsrvr_params = config['obsrvr']
actor_params = config['actor']
train_params = config['train']
cn_params = config['cn']
env_params = config['env']

with open(env_params['node_mapping_file'], 'r') as f:
    node2id = json.load(f)


class IBKG_Trainer:
    def __init__(self):

        self.device = torch.device(config['train']['device'] if torch.cuda.is_available() and config['train']['device'] == "cuda" else "cpu")
        
        self.ibkg = IBKG(
            max_nodes=train_params['max_nodes'],
            feat_dim=train_params['feat_dim'],
            rel2id=rel2id,
            node2id=node2id,
            hidden_dim=train_params['hidden_dim'],
            repr_dim=train_params['repr_dim'],
            latent_dim=train_params['latent_dim'],
            actor_model_name=actor_params['embedding_model'],
            actor_tokenizer_name=actor_params['tokenizer_model']
        )        

        location_model = TextModel(
            self.rootify_model(obsrvr_params['location_model'])[0],
            obsrvr_params['tokenizer_model'],
            self.rootify_model(obsrvr_params['location_model'])[1],     
        )

        surroundings_model = TextModel(
            self.rootify_model(obsrvr_params['surroundings_model'])[0],
            obsrvr_params['tokenizer_model'],
            self.rootify_model(obsrvr_params['surroundings_model'])[1],     
        )

        inventory_model = TextModel(
            self.rootify_model(obsrvr_params['inventory_model'])[0],
            obsrvr_params['tokenizer_model'],
            self.rootify_model(obsrvr_params['inventory_model'])[1],     
        )

        self.obsrvr = OBSRVR(
            location_model=location_model,
            surroundings_model=surroundings_model,
            inventory_model=inventory_model,
        )

        self.cn_injector = ConceptGraph(cn_params['concept_net_file'])

        self.env = TextEnv(env_params['rom_file'])

        self.optimizer = optim.Adam([
            {'params': self.ibkg.node_embedding.parameters()},
            {'params': self.ibkg.rgcn.parameters()},
            {'params': self.ibkg.attention.parameters()},
            {'params': self.ibkg.ib_encoder.parameters()},
            {'params': self.ibkg.prediction_encoder.parameters()},
            {'params': self.ibkg.action_decoder.parameters()}
        ], lr=train_params['learning_rate'])

        self.logger = Logger()

    def train(self, num_episodes):
        
        for episode in range(num_episodes):
            self.logger.log(f"Episode {episode + 1}/{num_episodes}")
            self.logger.progress_bar(episode + 1, num_episodes)

            initial_state = self.env.reset()
            observation = initial_state['observation']
            description = initial_state['description']
            inv_desc = initial_state['inventory']
            reward = initial_state['reward']
            done = initial_state['done']

            step_count = 0
            
            action=None
            location=None

            while not done:
                self.logger.log(f"Step {step_count + 1}")
                self.optimizer.zero_grad()

                observed_triplets = self.obsrvr.generate_triplets(
                    observation=observation,
                    description=description,
                    inventory=inv_desc,
                    previous_act=action,
                    previous_location=location
                )

                location = [tr[2] for tr in observed_triplets if tr[0] == 'you' and tr[1] == 'in'][0]

                self.ibkg.kg.update(observed_triplets)

                nodes = set([tr[0] for tr in observed_triplets]) | set([tr[2] for tr in observed_triplets])
                nodes = [node for node in nodes if node != 'you']

                injected_triplets = self.cn_injector.get_triplets_for_nodes(nodes)

                self.ibkg.kg.update(injected_triplets)

                valid_actions = self.env.get_admissible_actions()

                ib_loss, action, log_prob = self.ibkg.forward(
                    valid_actions=valid_actions,
                    beta=train_params['beta'],
                    epsilon=train_params['epsilon']
                )

                full_state = self.env.step(action)

                observation = full_state['observation']
                description = full_state['description']
                inv_desc = full_state['inventory']
                reward = full_state['reward']
                done = full_state['done']

                pg_loss = -log_prob * reward
                loss = ib_loss + pg_loss

                loss.backward()
                self.optimizer.step()

        with open(env_params['node_mapping_file'], 'w') as f:
            json.dump(self.ibkg.kg.node_mapping, f)


    def rootify_model(self, model_name):
        return '/'.join(model_name.split('/')[:2]), '/'.join(model_name.split('/')[2:]) if model_name.split('/')[2:] else None