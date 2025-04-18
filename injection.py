import json
from collections import defaultdict

class ConceptGraph:
    def __init__(self, filename):
        triplets = self.load_from_file(filename)
        self.graph = defaultdict(list)
        self.build_graph(triplets)

    def load_from_file(self, filename):
        with open(filename, 'r', encoding='utf-8') as file:
            raw_data = json.load(file)
        return [(d["subject"], d["relation"], d["object"]) for d in raw_data]
    
    def build_graph(self, triplets):
        self.triplets = triplets  
        for head, relation, tail in triplets:
            self.graph[head].append((head, relation, tail))
            self.graph[tail].append((head, relation, tail))  # optional, may keep if you want reverse lookups

    def get_neighbors(self, node):
        """Return only triplets where the node is the head (subject)."""
        return [t for t in self.graph.get(node, []) if t[0] == node]

    def get_triplets_for_nodes(self, nodes):
        """Return all triplets where each node in the list is the head."""
        results = []
        for node in nodes:
            results.extend(self.get_neighbors(node))
        return results
