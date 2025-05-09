import random
from graphviz import Graph

"""
Generate a random undirected tree with n nodes and render it using Graphviz.
"""

def generate_random_tree(n_nodes: int):
    """Return a list of edges for a random tree on n_nodes."""
    edges = []
    for i in range(1, n_nodes):
        parent = random.randrange(0, i)
        edges.append((parent, i))
    return edges

def render_tree(edges, filename='random_tree', fmt='png', view=True):
    """
    Build a Graphviz Graph, add nodes & edges, and render to file.
    If view=True, it will open the generated image.
    """
    dot = Graph('RandomTree', format=fmt)
    # optional: rankdir='TB' for top-to-bottom
    dot.attr(rankdir='TB', size='6,6')

    # add all nodes & edges
    for u, v in edges:
        dot.node(str(u))
        dot.node(str(v))
        dot.edge(str(u), str(v))

    output_path = dot.render(filename, cleanup=True)
    print(f"Rendered tree to {output_path}")
    return dot

N = 14
edges = generate_random_tree(N)
render_tree(edges, filename='random_tree', fmt='png', view=True)