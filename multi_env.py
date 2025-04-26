import jericho
import numpy as np
import multiprocessing
from jericho.template_action_generator import TemplateActionGenerator
from multiprocessing import Process, Pipe


def worker(remote, parent_remote, rom_path):
    """Worker function to run a single environment instance"""
    parent_remote.close()
    env = TextEnv(rom_path)
    
    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            action = data
            result = env.step(action)
            remote.send(result)
        elif cmd == 'reset':
            result = env.reset()
            remote.send(result)
        elif cmd == 'get_admissible_actions':
            valid_actions = env.get_admissible_actions()
            remote.send(valid_actions)
        elif cmd == 'close':
            remote.close()
            break
        else:
            raise NotImplementedError


class VectorizedTextEnv:
    """Vectorized version of TextEnv that runs multiple environments in parallel"""
    def __init__(self, rom_path, num_envs=4):
        self.num_envs = num_envs
        self.rom_path = rom_path
        
        # Create communication pipes
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(num_envs)])
        
        # Create and start worker processes
        self.processes = []
        for work_remote, remote in zip(self.work_remotes, self.remotes):
            process = Process(target=worker, args=(work_remote, remote, rom_path))
            process.daemon = True
            process.start()
            self.processes.append(process)
            work_remote.close()
    
    def reset(self):
        """Reset all environments"""
        for remote in self.remotes:
            remote.send(('reset', None))
        
        results = [remote.recv() for remote in self.remotes]
        return results
    
    def step(self, actions):
        """Take a step in each environment with the corresponding action"""
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        
        results = [remote.recv() for remote in self.remotes]
        return results
    
    def get_admissible_actions(self):
        """Get admissible actions for each environment"""
        for remote in self.remotes:
            remote.send(('get_admissible_actions', None))
        
        valid_actions = [remote.recv() for remote in self.remotes]
        return valid_actions
    
    def close(self):
        """Close all environments"""
        for remote in self.remotes:
            remote.send(('close', None))
        
        for process in self.processes:
            process.join()


class TextEnv:
    """Single environment implementation"""
    def __init__(self, rom_path):
        self.env = jericho.FrotzEnv(rom_path)
        self.state = None
        self.vocab = self.env.get_dictionary()
        self.vocab_mapping = {str(self.vocab[i]): i for i in range(len(self.vocab))}
        self.action_generator = TemplateActionGenerator(self.env.bindings)

    def reset(self):
        self.state, _ = self.env.reset()
        reward = 0
        done = False

        self.state = self.state.replace('\n', ' ')

        inv_desc, _, _, _ = self.env.step("inventory")
        inv_desc = inv_desc.replace('\n', ' ')

        description, _, _, _ = self.env.step("look")
        description = description.replace('\n', ' ')

        return {
            'observation': self.state,
            'description': description,
            'inventory': inv_desc,
            'reward': reward,
            'done': done
        }
        
    def step(self, action):
        self.state, reward, done, _ = self.env.step(action)
        self.state = self.state.replace('\n', ' ')

        inv_desc, _, _, _ = self.env.step("inventory")
        inv_desc = inv_desc.replace('\n', ' ')

        description, _, _, _ = self.env.step("look")
        description = description.replace('\n', ' ')

        return {
            'observation': self.state,
            'description': description,
            'inventory': inv_desc,
            'reward': reward,
            'done': done
        }
    
    def get_admissible_actions(self):
        valid_actions = self.env.get_valid_actions()
        
        if not valid_actions:
            return ["look", "inventory", "wait"]
            
        return valid_actions