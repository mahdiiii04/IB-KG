import jericho
from jericho.template_action_generator import TemplateActionGenerator


class TextEnv:
    def __init__(self, rom_path):
        self.env = jericho.FrotzEnv(rom_path)
        self.state = None
        self.vocab = self.env.get_dictionary()
        self.vocab_mapping = {str(self.vocab[i]): i for i in range(len(self.vocab))}
        self.action_generator =  TemplateActionGenerator(self.env.bindings)

    def reset(self):
        self.state, _= self.env.reset()
        reward = 0
        done = False

        self.state = self.state.replace('\n', ' ')

        inv_desc, _, _, _ = self.env.step("inventory")
        inv_desc = inv_desc.replace('\n', ' ')

        description, _, _, _ = self.env.step("look")
        description = description.replace('\n', ' ')

        return  {
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

        return  {
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