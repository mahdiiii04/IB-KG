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

        self.state.replace('\n', ' ')

        inv_desc, _, _, _ = self.env.step("inventory")
        inv_desc.replace('\n', ' ')

        description, _, _, _ = self.env.step("look")
        description.replace('\n', ' ')

        return  {
            'observation': self.state,
            'description': description,
            'inventory': inv_desc,
            'reward': reward,
            'done': done
        }
        
    def step(self, action):
        
        self.state, reward, done, _ = self.env.step(action)
        self.state.replace('\n', ' ')

        inv_desc, _, _, _ = self.env.step("inventory")
        inv_desc.replace('\n', ' ')

        description, _, _, _ = self.env.step("look")
        description.replace('\n', ' ')

        return  {
            'observation': self.state,
            'description': description,
            'inventory': inv_desc,
            'reward': reward,
            'done': done
        }
    
    def get_admissible_actions(self, objs):
        objs_ids = [self.vocab_mapping[obj] for obj in objs if obj in self.vocab_mapping]
        possible_actions = self.action_generator.generate_template_actions(objs, objs_ids)
        admissbile = self.env.find_valid_actions(possible_actions)
        return admissbile