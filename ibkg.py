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
            {'params': self.ibkg.action_decoder.parameters()}
        ], lr=train_params['learning_rate'])

        self.logger = Logger()

    def train(self, num_episodes):
        import time
        from datetime import timedelta
        import numpy as np
        import json
        
        # Metrics tracking
        episode_rewards = []
        episode_losses = []
        episode_steps = []
        episode_times = []
        total_actions_taken = 0
        
        self.logger.info(f"Starting training with {num_episodes} episodes")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Beta: {train_params['beta']}, Epsilon: {train_params['epsilon']}")
        
        start_time = time.time()
        
        for episode in range(num_episodes):
            episode_start = time.time()
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
            episode_loss_sum = 0
            
            action = None
            location = None
            
            # Log initial observation
            self.logger.info(f"Initial location: {description[:50]}...")
            self.logger.info(f"Initial inventory: {inv_desc[:50]}...")
    
            while not done and step_count < train_params.get('max_steps_per_episode', 100):
                step_start = time.time()
                self.logger.info(f"\n--- Step {step_count + 1} ---")
                self.optimizer.zero_grad()
    
                observed_triplets = self.obsrvr.generate_triplets(
                    observation=observation,
                    description=description,
                    inv_desc=inv_desc,
                    previous_act=action,
                    previous_location=location
                )
            
                # Log triplet count
                self.logger.info(f"Observed triplets: {len(observed_triplets)}")
                
                # Find current location
                location_triplets = [tr for tr in observed_triplets if tr[0] == 'you' and tr[1] == 'in']
                if location_triplets:
                    location = location_triplets[0][2]
                    self.logger.info(f"Current location: {location}")
                
                # Update knowledge graph
                prev_kg_size = len(self.ibkg.kg.triplets)
                self.ibkg.kg.update(observed_triplets)
                new_kg_size = len(self.ibkg.kg.triplets)
                
                self.logger.info(f"Knowledge graph: {prev_kg_size} → {new_kg_size} triplets")
    
                # Get nodes for concept injection
                nodes = set([tr[0] for tr in observed_triplets]) | set([tr[2] for tr in observed_triplets])
                nodes = [node for node in nodes if node != 'you']
    
                # Inject external knowledge
                injected_triplets = self.cn_injector.get_triplets_for_nodes(nodes)
                if injected_triplets:
                    self.logger.info(f"Injected {len(injected_triplets)} triplets from ConceptNet")
                    self.ibkg.kg.update(injected_triplets)
                    final_kg_size = len(self.ibkg.kg.triplets)
                    self.logger.info(f"Knowledge graph after injection: {final_kg_size} triplets")
    
                # Get valid actions
                valid_actions = self.env.get_admissible_actions()
                self.logger.info(f"Valid actions: {len(valid_actions)}")
                if len(valid_actions) < 5:  # Log all if there are just a few
                    self.logger.info(f"Actions: {', '.join(valid_actions)}")
                else:
                    self.logger.info(f"Sample actions: {', '.join(valid_actions[:3])}...")
    
                # Forward pass
                ib_loss, action, log_prob = self.ibkg.forward(
                    valid_actions=valid_actions,
                    beta=train_params['beta'],
                    epsilon=train_params['epsilon']
                )

                self.logger.info(f"Selected action: '{action}'")
                self.logger.info(f"Information bottleneck loss: {ib_loss.item():.4f}")
    
                # Take action in environment
                full_state = self.env.step(action)
                observation = full_state['observation']
                description = full_state['description']
                inv_desc = full_state['inventory']
                reward = full_state['reward']
                done = full_state['done']
                
                episode_reward += reward
                self.logger.info(f"Reward: {reward:.2f}, Cumulative: {episode_reward:.2f}")
                
                if reward != 0:
                    self.logger.info(f"Got non-zero reward! {reward:.2f}")
            
                # Policy gradient loss
                pg_loss = -log_prob * reward
                loss = ib_loss + pg_loss
                episode_loss_sum += loss.item()
                
                self.logger.info(f"Total loss: {loss.item():.4f}")
    
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.ibkg.parameters(), max_norm=1.0)
                self.optimizer.step()
                
                step_count += 1
                total_actions_taken += 1
                
                step_time = time.time() - step_start
                self.logger.info(f"Step time: {step_time:.2f}s")
                
                if done:
                    self.logger.info(f"Episode complete!")
                    
                # Print a snippet of the observation if it's not too long
                if len(observation) < 100:
                    self.logger.info(f"Observation: {observation}")
                else:
                    self.logger.info(f"Observation: {observation[:97]}...")
            
            # End of episode stats
            episode_time = time.time() - episode_start
            episode_times.append(episode_time)
            episode_rewards.append(episode_reward)
            episode_losses.append(episode_loss_sum / max(1, step_count))
            episode_steps.append(step_count)
            
            # Log episode summary
            self.logger.info(f"\n{'='*50}")
            self.logger.info(f"Episode {episode + 1} Summary:")
            self.logger.info(f"Steps: {step_count}")
            self.logger.info(f"Total reward: {episode_reward:.2f}")
            self.logger.info(f"Average loss: {episode_losses[-1]:.4f}")
            self.logger.info(f"Time taken: {timedelta(seconds=int(episode_time))}")
            
            # Log rolling statistics every 10 episodes
            if episode > 0 and (episode + 1) % 10 == 0:
                window = min(10, episode + 1)
                self.logger.info(f"\n{'*'*50}")
                self.logger.info(f"Statistics over last {window} episodes:")
                self.logger.info(f"Average reward: {np.mean(episode_rewards[-window:]):.2f}")
                self.logger.info(f"Average steps: {np.mean(episode_steps[-window:]):.1f}")
                self.logger.info(f"Average loss: {np.mean(episode_losses[-window:]):.4f}")
                self.logger.info(f"Average time per episode: {np.mean(episode_times[-window:]):.2f}s")
            
                # Save checkpoints periodically
                if hasattr(self, 'save_checkpoint') and callable(getattr(self, 'save_checkpoint')):
                    checkpoint_path = f"checkpoints/checkpoint_ep{episode+1}.pt"
                    self.save_checkpoint(checkpoint_path)
                    self.logger.info(f"Saved checkpoint to {checkpoint_path}")
            
            # Save the knowledge graph state periodically
            if (episode + 1) % 50 == 0:
                graph_state = {
                    "nodes": list(self.ibkg.kg.node_mapping.keys()),
                    "triplets": self.ibkg.kg.triplets,
                    "locations": self.ibkg.kg.locations
                }
                with open(f"logs/kg_state_ep{episode+1}.json", "w") as f:
                    json.dump(graph_state, f, indent=2)
                self.logger.info(f"Saved knowledge graph state to logs/kg_state_ep{episode+1}.json")
        
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
            "config": {
                "beta": train_params['beta'],
                "epsilon": train_params['epsilon'],
                "learning_rate": train_params['learning_rate']
            }
        }
        
        with open("logs/training_results.json", "w") as f:
            json.dump(results, f, indent=2)
        
        self.logger.info(f"Results saved to logs/training_results.json")


    def rootify_model(self, model_name):
        return '/'.join(model_name.split('/')[:2]), '/'.join(model_name.split('/')[2:]) if model_name.split('/')[2:] else None
