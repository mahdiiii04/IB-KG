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
        # Ensure log directories exist
        os.makedirs("logs", exist_ok=True)
        os.makedirs("checkpoints", exist_ok=True)
        
        self.timestamp = time.strftime("%Y%m%d-%H%M%S")
        self.log_dir = f"logs/run_{self.timestamp}"
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Create run-specific kg_states directory
        os.makedirs(f"{self.log_dir}/kg_states", exist_ok=True)
        
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
            actor_tokenizer_name=actor_params['tokenizer_model'],
            device=self.device
        )        

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

        self.env = TextEnv(env_params['rom_file'])

        self.optimizer = optim.Adam([
            {'params': self.ibkg.node_embedding.parameters()},
            {'params': self.ibkg.rgcn.parameters()},
            {'params': self.ibkg.attention.parameters()},
            {'params': self.ibkg.ib_encoder.parameters()},
            {'params': self.ibkg.prediction_encoder.parameters()},
            {'params': self.ibkg.action_decoder.parameters()},
            {'params': self.ibkg.critic.parameters()}
        ], lr=train_params['learning_rate'])

        self.logger = Logger(log_dir=self.log_dir, log_filename="training.log")
        
        # Save configuration for this run
        with open(f"{self.log_dir}/config.yaml", 'w') as f:
            yaml.dump(config, f)
        
        # Log initialization information
        self.logger.info(f"{'='*60}")
        self.logger.info(f"IBKG Training initialized at {self.timestamp}")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Configuration saved to {self.log_dir}/config.yaml")
        self.logger.info(f"{'='*60}")

    def save_checkpoint(self, filename):
        """Save model checkpoint"""
        checkpoint_path = f"{self.log_dir}/{filename}"
        torch.save({
            'ibkg_state_dict': self.ibkg.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'node_mapping': self.ibkg.kg.node_mapping
        }, checkpoint_path)
        return checkpoint_path

    def train(self, num_episodes, log_steps=False, log_obs=False, checkpoint_interval=10, kg_save_interval=50, injection=False):
        """
        Train the IBKG model
        
        Args:
            num_episodes: Number of episodes to train for
            log_steps: Whether to log detailed step information
            log_obs: Whether to log observations
            checkpoint_interval: How often to save checkpoints (in episodes)
            kg_save_interval: How often to save KG states (in episodes)
        """
        # Metrics tracking
        episode_rewards = []
        episode_losses = []
        episode_steps = []
        episode_times = []
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
        
        self.logger.info(f"Starting training with {num_episodes} episodes")
        self.logger.info(f"Logs saved to {self.log_dir}")
        
        start_time = time.time()
        
        for episode in range(num_episodes):
            episode_start = time.time()
            episode_metrics = {
                'episode': episode + 1,
                'rewards': [],
                'kg_size': [],
                'losses': {'ib': [], 'policy': [], 'value': []},
                'actions': []
            }

            self.logger.info(f"\n{'='*50}")
            self.logger.info(f"Episode {episode + 1}/{num_episodes}")
            self.logger.progress_bar(episode + 1, num_episodes)
    
            initial_state = self.env.reset()
            self.ibkg.kg.reset()
            observation = initial_state['observation']
            description = initial_state['description']
            inv_desc = initial_state['inventory']
            reward = initial_state['reward']
            done = initial_state['done']
    
            step_count = 0
            episode_reward = 0
            
            action = None
            location = None

            if log_obs:
                self.logger.info(f"Initial location: {description}")
                self.logger.info(f"Initial inventory: {inv_desc}")

            rewards = []
            values = []
            dones = []
            log_probs = []
            visited_locations = []
            
            # Log initial KG state
            episode_metrics['kg_size'].append(len(self.ibkg.kg.triplets))
    
            while not done and step_count < train_params.get('max_steps_per_episode', 100):
                step_start = time.time()

                if log_steps:
                    self.logger.info(f"\n--- Step {step_count + 1} ---")
                
                self.optimizer.zero_grad()
    
                observed_triplets = self.obsrvr.generate_triplets(
                    observation=observation,
                    description=description,
                    inv_desc=inv_desc,
                    previous_act=action,
                    previous_location=location
                )
            
                location_triplets = [tr for tr in observed_triplets if tr[0] == 'you' and tr[1] == 'in']
                if location_triplets:
                    location = location_triplets[0][2]

                    if log_obs:
                        self.logger.info(f"Current location: {location}")

                intrinsic_reward = 0

                if location not in visited_locations:
                    intrinsic_reward += 0.6
                    visited_locations.append(location)

                new_triplets = 0
                for triplet in observed_triplets:
                    if triplet not in self.ibkg.kg.triplets:
                        new_triplets += 1

                intrinsic_reward += new_triplets * 0.2
                
                prev_kg_size = len(self.ibkg.kg.triplets)
                self.ibkg.kg.update(observed_triplets)
                new_kg_size = len(self.ibkg.kg.triplets)

                
                if log_steps and prev_kg_size != new_kg_size:
                    self.logger.info(f"KG growth: {prev_kg_size} → {new_kg_size} (+{new_kg_size - prev_kg_size} triplets)")
                
                episode_metrics['kg_size'].append(new_kg_size)
    
                nodes = set([tr[0] for tr in observed_triplets]) | set([tr[2] for tr in observed_triplets])
                nodes = [node for node in nodes if node != 'you']
    
                injected_triplets = self.cn_injector.get_triplets_for_nodes(nodes)
                if injected_triplets and injection:
                    prev_kg_size = len(self.ibkg.kg.triplets)
                    self.ibkg.kg.update(injected_triplets)
                    new_kg_size = len(self.ibkg.kg.triplets)
                    
                    if log_steps and prev_kg_size != new_kg_size:
                        self.logger.info(f"KG injection: +{new_kg_size - prev_kg_size} triplets from ConceptNet")
    
                # Get valid actions
                valid_actions = self.env.get_admissible_actions()
                if log_steps:
                    self.logger.info(f"Valid actions ({len(valid_actions)}): {', '.join(valid_actions)}")

                # Forward pass
                progress = episode / num_episodes
                beta = train_params['beta_start'] + (train_params['beta_end'] - train_params['beta_end']) * progress
                epsilon = train_params['epsilon_start'] * ((1 - progress) ** 2) + train_params['epsilon_end'] * (1 - (1 - progress) ** 2)
                ib_loss, action, log_prob, value = self.ibkg.forward(
                    valid_actions=valid_actions,
                    beta=beta,
                    epsilon=epsilon
                )
                
                episode_metrics['actions'].append(action)
                episode_metrics['losses']['ib'].append(ib_loss.item())

                if log_steps:
                    self.logger.info(f"Selected action: '{action}'")
                    self.logger.info(f"Information bottleneck loss: {ib_loss.item():.4f}")
    
                # Take action in environment
                full_state = self.env.step(action)
                observation = full_state['observation']
                description = full_state['description']
                inv_desc = full_state['inventory']
                reward = full_state['reward']
                done = full_state['done']
                
                episode_metrics['rewards'].append(reward)

                intrinsic_decay = max(0.1, 1.0 - (episode / num_episodes))
                total_reward = reward + intrinsic_reward * intrinsic_decay

                rewards.append(total_reward)
                values.append(value)
                dones.append(done)
                log_probs.append(log_prob)
                
                episode_reward += reward
                if log_steps:
                    self.logger.info(f"Reward: {reward:.2f}, Cumulative: {episode_reward:.2f}")
                
                step_count += 1
                total_actions_taken += 1
                
                step_time = time.time() - step_start

                if log_steps:
                    self.logger.info(f"Step time: {step_time:.2f}s")
                
                if done and log_steps:
                    self.logger.info(f"Episode complete!")
                                    
                if log_obs:
                    self.logger.info(f"Observation: {observation}")
                

            advantages = self.compute_gae(rewards, values, dones)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            values = torch.stack(values)
            log_probs = torch.stack(log_probs)

            value_targets = (advantages + values).detach()
            value_loss = F.mse_loss(values, value_targets)
            
            policy_loss = (-log_probs * advantages).mean()

            loss = ib_loss + policy_loss + 0.5 * value_loss
            
            episode_metrics['losses']['policy'] = policy_loss.item()
            episode_metrics['losses']['value'] = value_loss.item()
            
            metrics['ib_losses'].append(float(ib_loss.item()))
            metrics['policy_losses'].append(float(policy_loss.item()))
            metrics['value_losses'].append(float(value_loss.item()))
            metrics['kg_sizes'].append(new_kg_size)
            
            episode_losses.append(loss.item())

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.ibkg.parameters(), max_norm=5.0)
            self.optimizer.step()
            
            # End of episode stats
            episode_time = time.time() - episode_start
            episode_times.append(episode_time)
            episode_rewards.append(episode_reward)
            episode_steps.append(step_count)
            
            metrics['reward_progression'].append(float(episode_reward))
            
            # Save episode metrics to JSONL file for real-time analysis
            with open(metrics_file, 'a') as f:
                f.write(json.dumps({
                    'episode': episode + 1,
                    'reward': episode_reward,
                    'steps': step_count,
                    'time': episode_time,
                    'kg_size': new_kg_size,
                    'losses': {
                        'total': loss.item(),
                        'ib': ib_loss.item(),
                        'policy': policy_loss.item(),
                        'value': value_loss.item()
                    }
                }) + '\n')
            
            self.logger.info(f"\n{'='*50}")
            self.logger.info(f"Episode {episode + 1} Summary:")
            self.logger.info(f"Steps: {step_count}")
            self.logger.info(f"Total reward: {episode_reward:.2f}")
            self.logger.info(f"Time taken: {timedelta(seconds=int(episode_time))}")
            self.logger.info(f"Losses - IB: {ib_loss.item():.4f}, Policy: {policy_loss.item():.4f}, Value: {value_loss.item():.4f}")
            self.logger.info(f"KG size: {new_kg_size} triplets")
            
            # Log rolling statistics every checkpoint_interval episodes
            if episode > 0 and (episode + 1) % checkpoint_interval == 0:
                window = min(checkpoint_interval, episode + 1)
                self.logger.info(f"\n{'*'*50}")
                self.logger.info(f"Statistics over last {window} episodes:")
                self.logger.info(f"Average reward: {np.mean(episode_rewards[-window:]):.2f}")
                self.logger.info(f"Average steps: {np.mean(episode_steps[-window:]):.1f}")
                self.logger.info(f"Average time per episode: {np.mean(episode_times[-window:]):.2f}s")
                self.logger.info(f"Average losses - IB: {np.mean(metrics['ib_losses'][-window:]):.4f}, " 
                                f"Policy: {np.mean(metrics['policy_losses'][-window:]):.4f}, "
                                f"Value: {np.mean(metrics['value_losses'][-window:]):.4f}")
            
                # Save checkpoints 
                checkpoint_path = self.save_checkpoint(f"checkpoint_ep{episode+1}.pt")
                self.logger.info(f"Saved checkpoint to {checkpoint_path}")
            
            # Save the knowledge graph state periodically
            if (episode + 1) % kg_save_interval == 0:
                graph_state = {
                    "nodes": list(self.ibkg.kg.node_mapping.keys()),
                    "triplets": self.ibkg.kg.triplets,
                    "locations": self.ibkg.kg.locations
                }
                kg_path = f"{self.log_dir}/kg_states/kg_state_ep{episode+1}.json"
                with open(kg_path, "w") as f:
                    json.dump(graph_state, f, indent=2)
                self.logger.info(f"Saved knowledge graph state to {kg_path}")
        
            # Save node mapping at each episode
            with open(env_params['node_mapping_file'], 'w') as f:
                json.dump(self.ibkg.kg.node_mapping, f)
        
        total_time = time.time() - start_time
        
        # Final stats
        self.logger.info(f"\n{'#'*60}")
        self.logger.info(f"Training completed in {timedelta(seconds=int(total_time))}")
        self.logger.info(f"Total episodes: {num_episodes}")
        self.logger.info(f"Total actions: {total_actions_taken}")
        self.logger.info(f"Average steps per episode: {np.mean(episode_steps):.1f}")
        self.logger.info(f"Average reward per episode: {np.mean(episode_rewards):.2f}")
        self.logger.info(f"Best episode reward: {np.max(episode_rewards):.2f} (Episode {np.argmax(episode_rewards)+1})")
        
        # Save final results
        results = {
            "episode_rewards": episode_rewards,
            "episode_losses": episode_losses,
            "episode_steps": episode_steps,
            "episode_times": episode_times,
            "total_time": total_time,
            "metrics": metrics,
            "config": {
                "beta": train_params['beta'],
                "epsilon": train_params['epsilon'],
                "learning_rate": train_params['learning_rate']
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
        return '/'.join(model_name.split('/')[:2]), '/'.join(model_name.split('/')[2:]) if model_name.split('/')[2:] else None

    def compute_gae(self, rewards, values, dones, gamma=0.99, lambda_=0.95):
        advantages = []
        advantage = 0  
        next_value = 0  
        
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
            
            advantage = delta + gamma * lambda_ * (1 - dones[t]) * advantage
            
            advantages.append(advantage)
            
            next_value = values[t]
        
        return torch.tensor(advantages[::-1], device=self.device)