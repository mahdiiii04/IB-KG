from pyvis.network import Network

class Grapher:
    def __init__(self, node_mapping, rel_mapping):
        self.triplets = []
        self.locations = []
        self.node_mapping = node_mapping
        self.rel_mapping = rel_mapping

    def reset(self):
        self.triplets = []
        self.locations = []
     
    def update(self, triplets):
        '''
        add the triplets to the graph
        triplets should be in the format (head, relation, tail)
        '''

        self.triplets = [t for t in self.triplets if t[0] != 'you']

        for triplet in triplets:
            head, relation, tail = triplet
            if triplet not in self.triplets:
                self.triplets.append(triplet)

                if head == 'you' and relation == 'in' and tail not in self.locations:
                    self.locations.append(tail)
                
                if head not in self.node_mapping:
                    self.node_mapping[head] = len(self.node_mapping)
                if tail not in self.node_mapping:
                    self.node_mapping[tail] = len(self.node_mapping)


    def get_nodes(self):
        nodes = set([t[0] for t in self.triplets] + [t[2] for t in self.triplets])
        return nodes
    
    def get_relations(self):
        relations = set([t[1] for t in self.triplets])
        return relations
    
    def get_sources_mapped(self):
        return [self.node_mapping[h] for h, _, _ in self.triplets]
    
    def get_destinations_mapped(self):
        return [self.node_mapping[t] for _, _, t in self.triplets]
    
    def get_relations_mapped(self):
        return [self.rel_mapping[r] for _, r, _ in self.triplets]
    
    def get_nodes_mapped(self):
        return [self.node_mapping[n] for n in self.get_nodes()]
    
    def build_graph(self):
        import dgl
        import torch

        src = self.get_sources_mapped()
        dst = self.get_destinations_mapped()
        rel = self.get_relations_mapped()

        num_nodes = len(self.node_mapping)

        g = dgl.graph((src, dst), num_nodes=num_nodes)
        g.edata['rel_type'] = torch.tensor(rel)

        return g


    def draw(self):
        net = Network(directed=True)

        # Add location nodes seperatly for coloring
        for loc in self.locations:
            net.add_node(loc, shape='circle', margin=20, color='#FFEB3B', font={'color': '#000000', 'size': 20})

        # Add the player node
        net.add_node('you', shape='circle', margin=20, color='#FFFFFF', font={'color': '#333333', 'size': 20})

        edge_style = {'color': '#888888', 'font': {'color': '#444444', 'size': 16}, 'width': 2, 'smooth': {'enabled': True, 'type': 'continuous'}}
        # Add the rest of nodes
        for triplet in self.triplets:
            head, relation, tail = triplet

            # if head not already in network, then it's not a location or inventory object
            if head not in net.nodes:
                net.add_node(head, shape='circle', margin=20, color='#B0BEC5', font={'color': '#000000', 'size': 20})  # Light gray-blue
            # if tail not already in network, then it's not a location or location object
            if tail not in net.nodes:
                net.add_node(tail, shape='circle', margin=20, color='#2196F3', font={'color': 'white', 'size': 20})
            
            net.add_edge(head, tail, label=relation, **edge_style)

        net.set_options("""
            {
            "physics": {
                "forceAtlas2Based": {
                "gravitationalConstant": -50,
                "centralGravity": 0.01,
                "springLength": 100,
                "springConstant": 0.08
                },
                "minVelocity": 0.75,
                "solver": "forceAtlas2Based"
            },
            "layout": {
                "randomSeed": 2,
                "improvedLayout": true
            }
            }
            """)


        return net.generate_html()

