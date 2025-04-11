class Grapher:
    def __init__(self):

        self.triplets = []

        
    def update(self, triplets):
        '''
        add the triplets to the graph
        triplets should be in the format (head, relation, tail)
        '''

        self.triplets = [t for t in self.triplets if t[0] != 'you']

        for triplet in triplets:
            if triplet not in self.triplets:
                self.triplets.append(triplet)


    def get_nodes(self):

        nodes = set([t[0] for t in self.triplets] + [t[2] for t in self.triplets])
        return nodes
    

    def get_relations(self):

        relations = set([t[1] for t in self.triplets])
        return relations
            