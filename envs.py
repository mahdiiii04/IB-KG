import jericho

class TextEnv:
    def __init__(self, rom_path):
        self.env = jericho.FrotzEnv(rom_path)
        self.state = None

    def reset(self):
        self.state = self.env.reset()
        return self.state
        
    def step(self, action):
        
        self.state, reward, done, _ = self.env.step(action)

        inv_desc, _, _, _ = self.env.step("inventory")

        description, _, _, _ = self.env.step("look")

        return  {
            'observation': self.state,
            'description': description,
            'inventory': inv_desc,
            'reward': reward,
            'done': done
        }