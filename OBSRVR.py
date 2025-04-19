import re

class OBSRVR:

    def __init__(self, location_model, inventory_model, surroundings_model):
        #Initilialize the three used models
        self.location_model = location_model
        self.inventory_model = inventory_model
        self.surroundings_model = surroundings_model

    
    def generate_triplets(self, observation, description, inv_desc, previous_act, previous_location):

        # Inventory model expeted to return items in inventory seperated by |
        inv_items = self.inventory_model.generate_output(inv_desc.lower()).split('|')
        inv_triplets = [('you', 'have', item.lower()) for item in inv_items if item != '']

        location = self.location_model.generate_output(description.lower()).lower()
        location = self.extract_core_phrase(location)
        loc_triplet = [('you', 'in', location)]

        # Surroundings model expeted to return surrounding objects seperated by |
        surr_objects = self.surroundings_model.generate_output(description.lower()).split('|')
        surr_triplets = [(obj.lower(), 'in', location) for obj in surr_objects if obj != '']

        triplets = loc_triplet + inv_triplets + surr_triplets

        # If location changes, we add the direction we moved in
        if previous_location:
            if location != previous_location:
                direction = self.extract_direction(previous_act)
                if direction:
                    triplets.append((location, f'{direction}_of', previous_location))

        return triplets

    def extract_direction(self, action_text):
        action_text = action_text.lower().strip()

        match = re.search(r"\b(go|move|walk|head|climb|descend)?\s*(north|south|east|west|up|down|northeast|northwest|southeast|southwest|n|s|e|w|u|d|ne|nw|se|sw)\b", action_text)
        
        if match:
            dir_raw = match.group(2)

            normalization = {
                "n": "north", "s": "south", "e": "east", "w": "west",
                "u": "up", "d": "down",
                "ne": "northeast", "nw": "northwest",
                "se": "southeast", "sw": "southwest"
            }

            return normalization.get(dir_raw, dir_raw) 
        return None
    
    def extract_core_phrase(self, text):
        pattern = r"^(in|on|at|to|by|from|with|about|into|over|after|under|above|around)\s+(a|an|the)\s+"
        return re.sub(pattern, '', text, flags=re.IGNORECASE).strip()



