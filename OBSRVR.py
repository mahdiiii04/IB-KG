import re

class OBSRVR:

    def __init__(self, location_model, inventory_model, surroundings_model):
        # Initialize the three models
        self.location_model = location_model
        self.inventory_model = inventory_model
        self.surroundings_model = surroundings_model

        # Caches
        self.description_to_location_triplet = {}
        self.description_to_location_name = {}
        self.description_to_surroundings = {}
        self.invdesc_to_triplets = {}

    def generate_triplets(self, observation, description, inv_desc, previous_act, previous_location):
        description = description.lower()
        inv_desc = inv_desc.lower()

        # Inventory model (cached)
        if inv_desc in self.invdesc_to_triplets:
            inv_triplets = self.invdesc_to_triplets[inv_desc]
        else:
            inv_items = self.inventory_model.generate_output(inv_desc).split('|')
            inv_triplets = [('you', 'have', item.strip().lower()) for item in inv_items if item.strip() != '']
            self.invdesc_to_triplets[inv_desc] = inv_triplets

        # Location model (cached)
        if description in self.description_to_location_triplet:
            loc_triplet = self.description_to_location_triplet[description]
            location = self.description_to_location_name[description]
        else:
            location = self.location_model.generate_output(description).lower()
            location = self.extract_core_phrase(location)
            loc_triplet = [('you', 'in', location)]
            self.description_to_location_triplet[description] = loc_triplet
            self.description_to_location_name[description] = location

        # Surroundings model (cached)
        if description in self.description_to_surroundings:
            surr_triplets = self.description_to_surroundings[description]
        else:
            surr_objects = self.surroundings_model.generate_output(description).split('|')
            surr_triplets = [(obj.strip().lower(), 'in', location) for obj in surr_objects if obj.strip() != '']
            self.description_to_surroundings[description] = surr_triplets

        # Combine all triplets
        triplets = loc_triplet + inv_triplets + surr_triplets

        # If location changes, add direction info
        if previous_location:
            if location != previous_location:
                direction = self.extract_direction(previous_act)
                if direction:
                    triplets.append((location, f'{direction}_of', previous_location))

        return triplets

    def extract_direction(self, action_text):
        action_text = action_text.lower().strip()
        match = re.search(
            r"\b(go|move|walk|head|climb|descend)?\s*(north|south|east|west|up|down|northeast|northwest|southeast|southwest|n|s|e|w|u|d|ne|nw|se|sw)\b",
            action_text
        )
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
