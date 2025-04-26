import torch
import torch.nn as nn
import torch.optim as optim
import yaml
import json
import os
import time
from datetime import timedelta
import numpy as np
import torch.nn.functional as F
from models import *
from OBSRVR import OBSRVR
from injection import ConceptGraph
from multi_env import VectorizedTextEnv, TextEnv
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
    def __init__(self, num_envs=4):
        # Ensure log directories exist
        os.makedirs("logs", exist_ok=True)
        os.makedirs("checkpoints", exist_ok=True)
        
        self.timestamp = time.strftime("%Y%m%d-%H%M%S")
        self.log_dir = f"logs/run_{self.timestamp}"
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Create run-specific kg_states directory
        os.makedirs(f"{self.log_dir}/kg_states", exist_ok=True)
        
        self.device = torch.device(config['train']['device'] if torch.cuda.is_available() and config['train']['device'] == "cuda" else "cpu")
        
        # Number of parallel environments
        self.num_envs = num_envs
        
        # Create a shared model
        self.ibkg = IBKG(
            max_nodes=train_params['max_nodes'],
            feat_dim=train_params['feat_dim'],
            rel2id=rel2id,
            node2id=node2id,
            hidden_dim=train_params['hidden_dim'],
            repr_dim=train_params['repr_dim'],
            latent_dim=train_params['latent_dim'],
            actor_model_name=actor_params['embedding_model'],
            actor_tokenizer_name=actor_params['tokenizer_model'],
            device=self.device
        )        

        # Models for observation processing
        location_model = TextModel(
            self.rootify_model(obsrvr_params['location_model'])[0],
            obsrvr_params['tokenizer_model'],
            self.device,
            self.rootify_model(obsrvr_params['location_model'])[1],     
        )

        surroundings_model = TextModel(
            self.rootify_model(obsrvr_params['surroundings_model'])[0],
            obsrvr_params['tokenizer_model'],
            self.device,
            self.rootify_model(obsrvr_params['surroundings_model'])[1],     
        )

        inventory_model = TextModel(
            self.rootify_model(obsrvr_params['inventory_model'])[0],
            obsrvr_params['tokenizer_model'],
            self.device,
            self.rootify_model(obsrvr_params['inventory_model'])[1],     
        )

        self.obsrvr = OBSRVR(
            location_model=location_model,
            surroundings_model=surroundings_model,
            inventory_model=inventory_model,
        )

        self.cn_injector = ConceptGraph(cn_params['concept_net_file'])

        # Create parallel environments
        self.env = VectorizedTextEnv(env_params['rom_file'], num_envs=self.num_envs)
        
        # Create separate optimizers for policy and value networks
        self.policy_optimizer = optim.Adam([
            {'params': self.ibkg.node_embedding.parameters()},
            {'params': self.ibkg.rgcn.parameters()},
            {'params': self.ibkg.attention.parameters()},
            {'params': self.ibkg.ib_encoder.parameters()},
            {'params': self.ibkg.action_decoder.parameters()}
        ], lr=train_params['learning_rate'])
        
        self.value_optimizer = optim.Adam([
            {'params': self.ibkg.prediction_encoder.parameters()},
            {'params': self.ibkg.critic.parameters()}
        ], lr=train_params['learning_rate'])

        self.logger = Logger(log_dir=self.log_dir, log_filename="training.log")
        
        # Save configuration for this run
        with open(f"{self.log_dir}/config.yaml", 'w') as f:
            yaml.dump(config, f)
        
        # For reward normalization (one for each env)
        self.reward_normalizers = [RewardNormalizer() for _ in range(self.num_envs)]
        
        # Create KG states for each environment
        self.kg_states = [IBKG_State(self.ibkg.kg) for _ in range(self.num_envs)]
        
        # Log initialization information
        self.logger.info(f"{'='*60}")
        self.logger.info(f"IBKG Training initialized at {self.timestamp}")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Number of parallel environments: {self.num_envs}")
        self.logger.info(f"Configuration saved to {self.log_dir}/config.yaml")
        self.logger.info(f"{'='*60}")

    def save_checkpoint(self, filename):
        """Save model checkpoint"""
        checkpoint_path = f"{self.log_dir}/{filename}"
        torch.save({
            'ibkg_state_dict': self.ibkg.state_dict(),
            'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
            'value_optimizer_state_dict': self.value_optimizer.state_dict(),
            'node_mapping': self.ibkg.kg.node_mapping,
            'reward_normalizers': self.reward_normalizers
        }, checkpoint_path)
        return checkpoint_path
    
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        if not os.path.exists(checkpoint_path):
            self.logger.info(f"Checkpoint {checkpoint_path} does not exist, starting from scratch.")
            return
            
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.ibkg.load_state_dict(checkpoint['ibkg_state_dict'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
        self.value_optimizer.load_state_dict(checkpoint['value_optimizer_state_dict'])
        self.ibkg.kg.node_mapping = checkpoint['node_mapping']
        
        if 'reward_normalizers' in checkpoint:
            self.reward_normalizers = checkpoint['reward_normalizers']
            
        self.logger.info(f"Loaded checkpoint from {checkpoint_path}")

    def get_kg_state(self, kg):
        """Extract a representation of the current KG state"""
        # Build DGL graph
        graph = kg.build_graph()
        if hasattr(graph, 'to'):
            graph = graph.to(self.device)
        
        # Get node IDs and features
        node_ids = list(range(len(kg.node_mapping)))
        node_ids_tensor = torch.LongTensor(node_ids).to(self.device)
        node_feat = self.ibkg.node_embedding(node_ids_tensor)
        
        # Get relation types
        rel_types = torch.tensor(kg.get_relations_mapped(), device=self.device)
        
        return {
            'graph': graph,
            'node_feat': node_feat,
            'rel_types': rel_types
        }

    def ppo_update(self, batch_states, batch_actions, batch_old_log_probs, batch_returns, batch_advantages, clip_ratio):
        """
        Perform a PPO update on the policy and value function separately
        """
    
        # Storage for new computations
        new_log_probs = []
        values = []
        ib_losses = []
    
        for i in range(len(batch_states)):
            # Process graph through RGCN
            h_t = self.ibkg.rgcn.forward(
                graph=batch_states[i]['graph'], 
                feat=batch_states[i]['node_feat'], 
                etypes=batch_states[i]['rel_types']
            )
    
            # Graph-level representation
            graph_repr = self.ibkg.attention.forward(h_t)
    
            # Latent representations
            z_t, mu_z, logvar_z = self.ibkg.ib_encoder.forward(graph_repr)
            h_next, mu_h, logvar_h = self.ibkg.prediction_encoder.forward(graph_repr)
    
            # Information Bottleneck loss
            l_1 = self.ibkg.kl_divergence(mu_z, logvar_z)
            l_2 = self.ibkg.kl_divergence(mu_h, logvar_h)
            ib_loss = l_1 - train_params.get('beta', 1.0) * l_2 + train_params.get('ib_reg', 0.02) * l_2.pow(2)
            ib_losses.append(ib_loss)
    
            # New log probability for the taken action
            valid_actions = [batch_actions[i]]
            scores = self.ibkg.action_decoder.forward(valid_actions, z_t)
            new_log_prob = F.log_softmax(scores, dim=0)[0]
            new_log_probs.append(new_log_prob)
    
            # Value prediction
            value = self.ibkg.critic.forward(z_t.detach())
            values.append(value)
    
        # Stack across batch
        new_log_probs = torch.stack(new_log_probs)
        values = torch.stack(values)
        ib_loss = torch.stack(ib_losses).mean()
    
        # Policy ratio
        ratio = torch.exp(new_log_probs - batch_old_log_probs)
    
        # PPO objective
        clip_adv = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * batch_advantages
        policy_loss = -torch.min(ratio * batch_advantages, clip_adv).mean()
    
        # Value loss
        value_loss = F.mse_loss(values.squeeze().float(), batch_returns.float())
    
        ## ==== POLICY UPDATE ====
        self.policy_optimizer.zero_grad()
    
        policy_total_loss = policy_loss + 0.01 * ib_loss
        policy_total_loss.backward(retain_graph=True)
    
        torch.nn.utils.clip_grad_norm_(
            list(self.ibkg.node_embedding.parameters()) +
            list(self.ibkg.rgcn.parameters()) +
            list(self.ibkg.attention.parameters()) +
            list(self.ibkg.ib_encoder.parameters()) +
            list(self.ibkg.action_decoder.parameters()),
            max_norm=0.5
        )
    
        self.policy_optimizer.step()
    
        ## ==== VALUE UPDATE ====
        self.value_optimizer.zero_grad()
    
        value_loss.backward()
    
        torch.nn.utils.clip_grad_norm_(
            list(self.ibkg.prediction_encoder.parameters()) +
            list(self.ibkg.critic.parameters()),
            max_norm=0.5
        )
    
        self.value_optimizer.step()
    
        return policy_loss + value_loss + ib_loss

    def train_ppo(self, num_episodes, epochs=4, batch_size=16, clip_ratio=0.2, log_steps=False, log_obs=False, checkpoint_interval=10, kg_save_interval=50, injection=False):
        """
        Train the IBKG model using PPO with parallel environments
        
        Args:
            num_episodes: Number of episodes to train for (per environment)
            epochs: Number of optimization epochs per batch of experience 
            batch_size: Mini-batch size for optimization
            clip_ratio: PPO clipping parameter
            log_steps: Whether to log detailed step information
            log_obs: Whether to log observations
            checkpoint_interval: How often to save checkpoints (in episodes)
            kg_save_interval: How often to save KG states (in episodes)
            injection: Whether to use ConceptNet knowledge injection
        """
        # Metrics tracking
        episode_rewards = [[] for _ in range(self.num_envs)]
        episode_losses = []
        episode_steps = [[] for _ in range(self.num_envs)]
        episode_times = [[] for _ in range(self.num_envs)]
        total_actions_taken = 0
        metrics = {
            'ib_losses': [],
            'policy_losses': [],
            'value_losses': [],
            'kg_sizes': [],
            'reward_progression': []
        }
        
        # Create metrics file for real-time tracking
        metrics_file = f"{self.log_dir}/metrics.jsonl"
        
        self.logger.info(f"Starting PPO training with {num_episodes} episodes per environment ({num_episodes * self.num_envs} total)")
        self.logger.info(f"Logs saved to {self.log_dir}")
        
        start_time = time.time()
        
        # Reset all environments to start
        env_states = self.env.reset()
        
        # Initialize environment tracking variables
        actions = [None] * self.num_envs
        locations = [None] * self.num_envs
        observations = [state['observation'] for state in env_states]
        descriptions = [state['description'] for state in env_states]
        inv_descs = [state['inventory'] for state in env_states]
        dones = [state['done'] for state in env_states]
        
        # Reset KG states for each environment
        for i in range(self.num_envs):
            self.kg_states[i].reset()
        
        # Episode tracking for each environment
        env_episode_counts = [0] * self.num_envs
        env_step_counts = [0] * self.num_envs
        env_episode_rewards = [0.0] * self.num_envs
        env_visited_locations = [[] for _ in range(self.num_envs)]
        
        # Storage for experience collection
        all_states = []
        all_actions = []
        all_old_log_probs = []
        all_rewards = []
        all_values = []
        all_dones = []
        
        # Track which environments are currently active
        active_envs = list(range(self.num_envs))
        completed_episodes = 0
        
        episode_start_times = [time.time()] * self.num_envs
        
        while completed_episodes < num_episodes * self.num_envs:
            # Step through each active environment
            env_batch_states = []
            env_batch_actions = []
            env_batch_log_probs = []
            env_batch_values = []
            
            # Process observations and update KGs for each active environment
            for i in active_envs:
                # Process observation and update KG
                observed_triplets = self.obsrvr.generate_triplets(
                    observation=observations[i],
                    description=descriptions[i],
                    inv_desc=inv_descs[i],
                    previous_act=actions[i],
                    previous_location=locations[i]
                )
                
                # Extract location information
                location_triplets = [tr for tr in observed_triplets if tr[0] == 'you' and tr[1] == 'in']
                if location_triplets:
                    locations[i] = location_triplets[0][2]
                
                # Update KG with observed triplets
                self.kg_states[i].update(observed_triplets)
                
                # Knowledge injection from ConceptNet if enabled
                if injection:
                    nodes = set([tr[0] for tr in observed_triplets]) | set([tr[2] for tr in observed_triplets])
                    nodes = [node for node in nodes if node != 'you']
                    
                    injected_triplets = self.cn_injector.get_triplets_for_nodes(nodes)
                    if injected_triplets:
                        self.kg_states[i].update(injected_triplets)
                
                # Capture current KG state
                kg_state = self.get_kg_state(self.kg_states[i])
                env_batch_states.append(kg_state)
            
            # Get valid actions for each active environment
            valid_actions_list = self.env.get_admissible_actions()
            valid_actions_batch = [valid_actions_list[i] for i in active_envs]
            
            # Get actions from policy for each environment
            batch_actions = []
            batch_log_probs = []
            batch_values = []
            batch_ib_losses = []
            
            # Setup exploration parameters
            total_episodes = sum(env_episode_counts)
            progress = total_episodes / (num_episodes * self.num_envs)
            epsilon = train_params['epsilon_start'] * (1 - progress) + train_params['epsilon_end'] * progress
            beta = train_params['beta_start'] + (train_params['beta_end'] - train_params['beta_start']) * progress
            
            # Forward pass for each environment
            for env_idx, valid_actions in zip(active_envs, valid_actions_batch):
                with torch.no_grad():
                    ib_loss, action, log_prob, value = self.ibkg.forward(
                        valid_actions=valid_actions,
                        beta=beta,
                        epsilon=epsilon,
                        ib_reg=train_params.get('ib_reg', 0.02),
                        kg_state=env_batch_states[active_envs.index(env_idx)]
                    )
                
                batch_actions.append(action)
                batch_log_probs.append(log_prob)
                batch_values.append(value)
                batch_ib_losses.append(ib_loss)
                
                actions[env_idx] = action
            
            # Take actions in environments
            env_batch_actions = [actions[i] for i in active_envs]
            env_results = self.env.step(env_batch_actions)
            
            # Process results and compute rewards
            env_batch_rewards = []
            env_batch_dones = []
            new_active_envs = []
            
            for idx, env_idx in enumerate(active_envs):
                result = env_results[idx]
                observations[env_idx] = result['observation']
                descriptions[env_idx] = result['description']
                inv_descs[env_idx] = result['inventory']
                reward = result['reward']
                dones[env_idx] = result['done']
                
                # Calculate intrinsic reward for exploration
                intrinsic_reward = 0
                
                if locations[env_idx] and locations[env_idx] not in env_visited_locations[env_idx]:
                    intrinsic_reward += 0.3
                    env_visited_locations[env_idx].append(locations[env_idx])
                
                # Apply decreasing weight to intrinsic rewards over time
                intrinsic_decay = max(0.05, 0.5 - (0.4 * progress))
                total_reward = reward + intrinsic_reward * intrinsic_decay
                
                # Normalize rewards for stability
                normalized_reward = self.reward_normalizers[env_idx].normalize(total_reward)
                
                env_batch_rewards.append(normalized_reward)
                env_batch_dones.append(dones[env_idx])
                
                # Update episode statistics
                env_episode_rewards[env_idx] += reward
                env_step_counts[env_idx] += 1
                total_actions_taken += 1
                
                # Keep track of active environments
                if not dones[env_idx] and env_step_counts[env_idx] < train_params.get('max_steps_per_episode', 100):
                    new_active_envs.append(env_idx)
            
            # Store experience
            all_states.extend(env_batch_states)
            all_actions.extend(batch_actions)
            all_old_log_probs.extend(batch_log_probs)
            all_rewards.extend(env_batch_rewards)
            all_values.extend(batch_values)
            all_dones.extend(env_batch_dones)
            
            # Check if any environments have completed their episodes
            for env_idx in active_envs:
                if env_idx not in new_active_envs:
                    # Episode completed for this environment
                    episode_time = time.time() - episode_start_times[env_idx]
                    episode_times[env_idx].append(episode_time)
                    episode_rewards[env_idx].append(env_episode_rewards[env_idx])
                    episode_steps[env_idx].append(env_step_counts[env_idx])
                    
                    completed_episodes += 1
                    env_episode_counts[env_idx] += 1
                    
                    # Log episode completion
                    self.logger.info(f"Env {env_idx} completed episode {env_episode_counts[env_idx]} with reward {env_episode_rewards[env_idx]:.2f} in {env_step_counts[env_idx]} steps")
                    
                    # Save metrics
                    metrics['reward_progression'].append(float(env_episode_rewards[env_idx]))
                    metrics['kg_sizes'].append(len(self.kg_states[env_idx].triplets))
                    
                    with open(metrics_file, 'a') as f:
                        f.write(json.dumps({
                            'env': env_idx,
                            'episode': env_episode_counts[env_idx],
                            'reward': env_episode_rewards[env_idx],
                            'steps': env_step_counts[env_idx],
                            'time': episode_time,
                            'kg_size': len(self.kg_states[env_idx].triplets),
                            'epsilon': epsilon,
                            'beta': beta
                        }) + '\n')
                    
                    # Reset this environment
                    if completed_episodes < num_episodes * self.num_envs:
                        self.kg_states[env_idx].reset()
                        env_visited_locations[env_idx] = []
                        env_episode_rewards[env_idx] = 0.0
                        env_step_counts[env_idx] = 0
                        episode_start_times[env_idx] = time.time()
                        
                        # Reset environment
                        result = self.env.reset()[env_idx]
                        observations[env_idx] = result['observation']
                        descriptions[env_idx] = result['description']
                        inv_descs[env_idx] = result['inventory']
                        dones[env_idx] = result['done']
                        
                        # Add back to active environments
                        new_active_envs.append(env_idx)
            
            # Update active environments for next iteration
            active_envs = new_active_envs
            
            # Perform PPO updates when we have enough data or all environments are done
            if len(all_rewards) >= batch_size * 4 or not active_envs:
                if len(all_rewards) > 0:
                    # Calculate advantages and returns
                    rewards_tensor = torch.tensor(all_rewards, device=self.device)
                    values_tensor = torch.stack(all_values)
                    dones_tensor = torch.tensor(all_dones, device=self.device)
                    
                    # Calculate advantages using GAE
                    advantages = self.compute_gae(rewards_tensor, values_tensor, dones_tensor)
                    returns = advantages + values_tensor.squeeze()
                    
                    # Normalize advantages
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                    
                    # Stack log probs
                    old_log_probs_tensor = torch.stack(all_old_log_probs)
                    
                    # Multiple epochs of PPO updates
                    avg_loss = 0
                    for _ in range(epochs):
                        # Create random indices for mini-batches
                        indices = torch.randperm(len(all_rewards))
                        
                        for start_idx in range(0, len(all_rewards), batch_size):
                            end_idx = min(start_idx + batch_size, len(all_rewards))
                            batch_indices = indices[start_idx:end_idx]
                            
                            # Skip if batch is too small
                            if len(batch_indices) < 2:
                                continue
                            
                            # Extract batch data
                            batch_states = [all_states[i] for i in batch_indices]
                            batch_actions = [all_actions[i] for i in batch_indices]
                            batch_old_log_probs = old_log_probs_tensor[batch_indices]
                            batch_returns = returns[batch_indices]
                            batch_advantages = advantages[batch_indices]
                            
                            # Perform PPO update
                            loss = self.ppo_update(
                                batch_states, 
                                batch_actions,
                                batch_old_log_probs, 
                                batch_returns, 
                                batch_advantages,
                                clip_ratio
                            )
                            avg_loss += loss.item()
                    
                    # Calculate average loss over all updates
                    if epochs > 0 and batch_size > 0:
                        num_updates = max(1, epochs * ((len(all_rewards) + batch_size - 1) // batch_size))
                        avg_loss /= num_updates
                        episode_losses.append(avg_loss)
                    
                    # Clear experience buffers after update
                    all_states = []
                    all_actions = []
                    all_old_log_probs = []
                    all_rewards = []
                    all_values = []
                    all_dones = []
            
            # Save checkpoint periodically based on total episodes completed
            if completed_episodes > 0 and completed_episodes % checkpoint_interval == 0:
                checkpoint_path = self.save_checkpoint(f"checkpoint_ep{completed_episodes}.pt")
                checkpoint_path = self.save_checkpoint(f"checkpoint_ep{completed_episodes}.pt")
                self.logger.info(f"Checkpoint saved to {checkpoint_path}")
            
            # Save KG states periodically
            if completed_episodes > 0 and completed_episodes % kg_save_interval == 0:
                for env_idx in range(self.num_envs):
                    graph_state = {
                        "nodes": list(self.kg_states[env_idx].node_mapping.keys()),
                        "triplets": self.kg_states[env_idx].triplets,
                        "locations": self.kg_states[env_idx].locations
                    }
                    kg_path = f"{self.log_dir}/kg_states/kg_state_env{env_idx}_ep{completed_episodes}.json"
                    with open(kg_path, "w") as f:
                        json.dump(graph_state, f, indent=2)
                self.logger.info(f"Saved knowledge graph states to {self.log_dir}/kg_states/")
        
        # End of training - save final results
        total_time = time.time() - start_time
        
        # Final stats
        self.logger.info(f"\n{'#'*60}")
        self.logger.info(f"Training completed in {timedelta(seconds=int(total_time))}")
        self.logger.info(f"Total episodes: {completed_episodes}")
        self.logger.info(f"Total actions: {total_actions_taken}")
        
        # Calculate aggregate statistics across all environments
        all_rewards = [item for sublist in episode_rewards for item in sublist]
        all_steps = [item for sublist in episode_steps for item in sublist]
        all_times = [item for sublist in episode_times for item in sublist]
        
        self.logger.info(f"Average steps per episode: {np.mean(all_steps):.1f}")
        self.logger.info(f"Average reward per episode: {np.mean(all_rewards):.2f}")
        self.logger.info(f"Best episode reward: {np.max(all_rewards):.2f}")
        
        # Save final results
        results = {
            "episode_rewards": episode_rewards,
            "episode_losses": episode_losses if episode_losses else [],
            "episode_steps": episode_steps,
            "episode_times": [[float(t) for t in env_times] for env_times in episode_times],
            "total_time": float(total_time),
            "metrics": metrics,
            "config": {
                "beta_start": train_params['beta_start'],
                "beta_end": train_params['beta_end'],
                "epsilon_start": train_params['epsilon_start'],
                "epsilon_end": train_params['epsilon_end'],
                "learning_rate": train_params['learning_rate'],
                "num_envs": self.num_envs
            }
        }
        
        results_path = f"{self.log_dir}/training_results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        
        self.logger.info(f"Results saved to {results_path}")
        
        # Save final model checkpoint
        final_checkpoint = self.save_checkpoint("final_model.pt")
        self.logger.info(f"Final model saved to {final_checkpoint}")
        
        return results

    def rootify_model(self, model_name):
        """Helper to split model name into root path and specific model name"""
        return '/'.join(model_name.split('/')[:2]), '/'.join(model_name.split('/')[2:]) if model_name.split('/')[2:] else None

    def compute_gae(self, rewards, values, dones, gamma=0.99, lambda_=0.95):
        """Compute Generalized Advantage Estimation with death penalty adjustment"""
        advantages = torch.zeros_like(rewards)
        last_advantage = 0
        last_value = 0
    
        for t in reversed(range(len(rewards))):
            mask = 1.0 - dones[t].float()
            if t == len(rewards) - 1:
                next_value = last_value
            else:
                next_value = values[t + 1]
    
            delta = rewards[t] + gamma * next_value * mask - values[t]
            advantages[t] = delta + gamma * lambda_ * mask * last_advantage
    
            if dones[t] and rewards[t] < -5:
                advantages[t] -= 10  # punish death strongly
    
            last_advantage = advantages[t]
    
        return advantages


class RewardNormalizer:
    """Simple reward normalizer using running statistics"""
    def __init__(self, epsilon=1e-8):
        self.mean = 0
        self.std = 1
        self.count = 0
        self.epsilon = epsilon
        
    def normalize(self, reward):
        """Normalize a reward value"""
        self.count += 1
        # Update running mean and std
        delta = reward - self.mean
        self.mean += delta / self.count
        delta2 = reward - self.mean
        self.std = ((self.count - 1) * self.std**2 + delta * delta2) / self.count
        self.std = max(self.epsilon, np.sqrt(self.std))
        
        # Normalize the reward
        return float(reward - self.mean) / self.std


class IBKG_State:
    """Stateful representation of a Knowledge Graph for parallel environments"""
    def __init__(self, initial_kg=None):
        if initial_kg:
            self.triplets = initial_kg.triplets.copy()
            self.node_mapping = initial_kg.node_mapping.copy()
            self.locations = initial_kg.locations.copy() if hasattr(initial_kg, 'locations') else []
        else:
            self.triplets = []
            self.node_mapping = {}
            self.locations = []
    
    def reset(self):
        """Reset the knowledge graph state"""
        self.triplets = []
        self.locations = []
        # Keep node mapping to preserve node IDs
    
    def update(self, triplets):
        """Update the knowledge graph with new triplets"""
        for triplet in triplets:
            if triplet not in self.triplets:
                self.triplets.append(triplet)
                
                # Update node mapping
                for node in [triplet[0], triplet[2]]:
                    if node not in self.node_mapping:
                        self.node_mapping[node] = len(self.node_mapping)
                
                # Update locations
                if triplet[0] == 'you' and triplet[1] == 'in' and triplet[2] not in self.locations:
                    self.locations.append(triplet[2])
    
    def get_relations_mapped(self):
        """Get relation types with IDs"""
        return [rel2id.get(triplet[1], 0) for triplet in self.triplets]
    
    def build_graph(self):
        """Build a DGL graph from the triplets"""
        import dgl
        
        if not self.triplets:
            # Create an empty graph with a single node
            g = dgl.heterograph({('node', 'relation', 'node'): ([], [])})
            g.add_nodes(1)
            return g
        
        # Use known nodes to build graph
        src_nodes = [self.node_mapping.get(triplet[0], 0) for triplet in self.triplets]
        dst_nodes = [self.node_mapping.get(triplet[2], 0) for triplet in self.triplets]
        etypes = [rel2id.get(triplet[1], 0) for triplet in self.triplets]
        
        # Ensure we have at least one node
        num_nodes = max(len(self.node_mapping), 1)
        
        # Create heterograph
        g = dgl.graph((src_nodes, dst_nodes), num_nodes=num_nodes)
        g.edata['etype'] = torch.tensor(etypes)
        
        return g
    
    def clone(self):
        """Create a deep copy of this KG state"""
        clone = IBKG_State()
        clone.triplets = self.triplets.copy()
        clone.node_mapping = self.node_mapping.copy()
        clone.locations = self.locations.copy() if hasattr(self, 'locations') else []
        return clone