# Paths

• An Eulerian path is a path that goes through each edge exactly once.
• A Hamiltonian path is a path that visits each node exactly once.

The existence of Eulerian paths and circuits depends on the degrees of the nodes.
First, an undirected graph has an Eulerian path exactly when all the edges
belong to the same connected component and
• the degree of each node is even or
• the degree of exactly two nodes is odd, and the degree of all other nodes is
even.
In the first case, each Eulerian path is also an Eulerian circuit. In the second
case, the odd-degree nodes are the starting and ending nodes of an Eulerian path
which is not an Eulerian circuit.

In a directed graph, we focus on indegrees and outdegrees of the nodes. A
directed graph contains an Eulerian path exactly when all the edges belong to
the same connected component and
• in each node, the indegree equals the outdegree, or
• in one node, the indegree is one larger than the outdegree, in another node,
the outdegree is one larger than the indegree, and in all other nodes, the
indegree equals the outdegree.

# HAMILTONIANS

No efficient method is known for testing if a graph contains a Hamiltonian path,
and the problem is NP-hard. Still, in some special cases, we can be certain that a
graph contains a Hamiltonian path.
A simple observation is that if the graph is complete, i.e., there is an edge
between all pairs of nodes, it also contains a Hamiltonian path. Also stronger
results have been achieved:
• Dirac’s theorem: If the degree of each node is at least n/2, the graph
contains a Hamiltonian path.
• Ore’s theorem: If the sum of degrees of each non-adjacent pair of nodes is
at least n, the graph contains a Hamiltonian path.

A more efficient solution is based on dynamic programming (see Chapter
10.5). The idea is to calculate values of a function possible(S, x), where S is a
subset of nodes and x is one of the nodes. The function indicates whether there is
a Hamiltonian path that visits the nodes of S and ends at node x. It is possible to
implement this solution in O(2nn
2
) time