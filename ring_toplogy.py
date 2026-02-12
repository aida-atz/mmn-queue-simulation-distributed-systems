class RingTopology:
    def __init__(self, n):
        """Initialize the ring with n nodes."""
        self.nodes = {i: {'next': (i + 1) % n, 'prev': (i - 1) % n} for i in range(n)}

    def add_node(self):
        """Add a new node to the ring."""
        new_index = len(self.nodes)
        prev_index = new_index - 1
        first_index = 0
        self.nodes[new_index] = {'next': first_index, 'prev': prev_index}
        self.nodes[prev_index]['next'] = new_index
        self.nodes[first_index]['prev'] = new_index

    def remove_node(self, node):
        """Remove a node from the ring."""
        if node in self.nodes:
            next_node = self.nodes[node]['next']
            prev_node = self.nodes[node]['prev']
            self.nodes[prev_node]['next'] = next_node
            self.nodes[next_node]['prev'] = prev_node
            del self.nodes[node]

    def traverse_ring(self, start_node, d):
        """Traverse the ring starting from a node."""
        result = []
        current = start_node
        for _ in range(d):
            result.append(current)
            current = self.nodes[current]['next']
        return result