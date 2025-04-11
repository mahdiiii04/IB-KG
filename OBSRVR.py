from transformers import T5Tokenizer, T5ForConditionalGeneration
import re

class TextModel:
    def __init__(self, model_name, tokenizer_name):
        self.model_name = model_name
        self.tokenizer_name = tokenizer_name

        self.load_model()

    def load_model(self):
        self.tokenizer = T5Tokenizer.from_pretrained(self.tokenizer_name)
        self.model = T5ForConditionalGeneration.from_pretrained(self.model_name)


    def generate_output(self, input_text):
        input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids
        output_ids = self.model.generate(input_ids, max_length=50, num_beams=4)
        output_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return output_text



class OBSRVR:

    def __init__(self, location_model, inventory_model, surroundings_model, tokenizer):
        self.location_model = TextModel(location_model, tokenizer)
        self.inventory_model = TextModel(inventory_model, tokenizer)
        self.surroundings_model = TextModel(surroundings_model, tokenizer)

    
    def generate_triplets(self, observation, description, inv_desc, previous_act, previous_location):

        # Inventory model expeted to return items in inventory seperated by |
        inv_items = self.inventory_model.generate_output(inv_desc).split('|')
        inv_triplets = [('you', 'have', item) for item in inv_items if item != '']

        location = self.location_model.generate_output(description)
        loc_triplet = [('you', 'in', location)]

        surr_objects = self.surroundings_model.generate_output(description).split('|')
        surr_triplets = [(obj, 'in', location) for obj in surr_objects if obj != '']

        triplets = loc_triplet + inv_triplets + surr_triplets

        if location != previous_location:
            direction = self.extract_direction(previous_act)
            if direction:
                triplets.append((location, direction, previous_location))

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




