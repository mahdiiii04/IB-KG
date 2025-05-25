%%writefile demo.py
import torch
import torch.nn as nn
import yaml
import json
import random
import numpy as np

from models import NOIBKG, TextModel
from OBSRVR import OBSRVR
from injection import ConceptGraph
from env import TextEnv

try:
    # If available, use Hugging Face Hub to download the checkpoint
    from huggingface_hub import hf_hub_download
except ImportError:
    hf_hub_download = None

class DemoRunner:
    """
    DemoRunner sets up a text environment with the NOIBKG model and observers.
    Each call to advance() performs one step: it observes the state, updates the KG,
    selects an action via NOIBKG.forward, steps the environment, and returns results.
    """

    def __init__(self, hf_repo_id, seed=42):
        """
        Initializes models and environment.
        
        Args:
            hf_repo_id (str): Hugging Face repository ID containing the checkpoint file 'ibkg_model.pt'.
            seed (int): Random seed for reproducibility.
        """
        # Load configuration and mappings
        with open('conf/config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        with open('mapping/rel2id.json', 'r') as f:
            rel2id = json.load(f)
        env_params = config['env']
        with open(env_params['node_mapping_file'], 'r') as f:
            node2id = json.load(f)
        with open(env_params['action_mapping_file'], 'r') as f:
            act2id = json.load(f)
        
        # Set device and random seeds for reproducibility
        self.device = torch.device(config['train']['device'] if (torch.cuda.is_available() and config['train']['device'] == "cuda") else "cpu")
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        # Initialize NOIBKG model
        train_params = config['train']
        self.model = NOIBKG(
            max_nodes=train_params['max_nodes'],
            feat_dim=train_params['feat_dim'],
            rel2id=rel2id,
            node2id=node2id,
            act2id=act2id,
            hidden_dim=train_params['hidden_dim'],
            repr_dim=train_params['repr_dim'],
            device=self.device
        )
        
        # Load model weights from Hugging Face repository checkpoint
        if hf_repo_id is None:
            raise ValueError("hf_repo_id must be provided to load the model checkpoint.")
        if hf_hub_download is None:
            raise ImportError("huggingface_hub is not installed. Install it to load the model checkpoint.")
        
        # Download checkpoint file and load state
        checkpoint_path = hf_hub_download(repo_id=hf_repo_id, filename='ibkg_model.pt')
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['ibkg_state_dict'])
        # Restore knowledge graph and action mappings from checkpoint
        if 'node_mapping' in checkpoint:
            self.model.kg.node_mapping = checkpoint['node_mapping']
        if 'action_mapping' in checkpoint:
            self.model.action_decoder.encoder.action_mapping = checkpoint['action_mapping']
        
        # Initialize observer (OBSRVR) models
        obsrvr_params = config['obsrvr']
        def _rootify_model(name):
            parts = name.split('/')
            base = '/'.join(parts[:2])
            subfolder = '/'.join(parts[2:]) if len(parts) > 2 else None
            return base, subfolder
        
        loc_model_name, loc_sub = _rootify_model(obsrvr_params['location_model'])
        self.location_model = TextModel(loc_model_name, obsrvr_params['tokenizer_model'], self.device, loc_sub)
        sur_model_name, sur_sub = _rootify_model(obsrvr_params['surroundings_model'])
        self.surroundings_model = TextModel(sur_model_name, obsrvr_params['tokenizer_model'], self.device, sur_sub)
        inv_model_name, inv_sub = _rootify_model(obsrvr_params['inventory_model'])
        self.inventory_model = TextModel(inv_model_name, obsrvr_params['tokenizer_model'], self.device, inv_sub)
        self.obsrvr = OBSRVR(
            location_model=self.location_model,
            surroundings_model=self.surroundings_model,
            inventory_model=self.inventory_model
        )
        
        # ConceptNet injector (optional)
        self.cn = ConceptGraph(config['cn']['concept_net_file']) if 'cn' in config else None
        
        # Initialize environment
        self.env = TextEnv(env_params['rom_file'])
        # Reset environment and knowledge graph
        initial_state = self.env.reset()
        self.model.kg.reset()
        self.observation = initial_state.get('observation', "")
        self.description = initial_state.get('description', "")
        self.inventory = initial_state.get('inventory', "")
        self.done = initial_state.get('done', False)
        self.location = None
        self.last_action = None
    
    def advance(self):
        """
        Performs one step in the environment:
        - Observes current state with OBSRVR, updates the KG.
        - Selects an action via NOIBKG model (forward pass).
        - Steps the environment with that action.
        - Returns a dict with state and action info.
        """
        # If episode is done, optionally reset environment
        if self.done:
            state = self.env.reset()
            self.model.kg.reset()
            self.observation = state.get('observation', "")
            self.description = state.get('description', "")
            self.inventory = state.get('inventory', "")
            self.done = state.get('done', False)
            self.location = None
            self.last_action = None
        
        # Generate observed triplets from the state
        triplets = self.obsrvr.generate_triplets(
            observation=self.observation,
            description=self.description,
            inv_desc=self.inventory,
            previous_act=self.last_action,
            previous_location=self.location
        )
        # Update current location if any 'you in X' triplet
        for subj, rel, obj in triplets:
            if subj == 'you' and rel == 'in':
                self.location = obj
        # Update knowledge graph with new triplets
        self.model.kg.update(triplets)
        # (Optional) Inject ConceptNet knowledge if needed
        # if self.cn is not None:
        #     nodes = {n for tr in triplets for n in (tr[0], tr[2]) if tr[0] != 'you'}
        #     injected = self.cn.get_triplets_for_nodes(list(nodes))
        #     self.model.kg.update(injected)
        
        # Get valid actions from the environment
        valid_actions = self.env.get_admissible_actions()
        # Model forward pass to get action and probabilities (no exploration)
        action, log_prob, value, probs = self.model.forward(
            valid_actions,
            epsilon=0.0,
            temperature=1.0
        )
        # Step the environment with the chosen action
        full_state = self.env.step(action)
        # Update runner state
        self.observation = full_state.get('observation', "")
        self.description = full_state.get('description', "")
        self.inventory = full_state.get('inventory', "")
        reward = full_state.get('reward', 0)
        self.done = full_state.get('done', False)
        self.last_action = action
        
        # Prepare return dict
        result = {
            'description': self.description,
            'reward': reward,
            'valid_actions': valid_actions,
            'chosen_action': action,
            'probability_distribution': probs.detach().cpu().tolist(),
        }
        # Optionally include observation and inventory if present
        if 'observation' in full_state:
            result['observation'] = self.observation
        if 'inventory' in full_state:
            result['inventory'] = self.inventory
        
        return result