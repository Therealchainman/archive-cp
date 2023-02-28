# EULER TOUR TECHNIQUE

## DEFINITIONS

Euler Tour Technique to a random tree

relabel or index of the tree nodes


Necessary and sufficient conditions

An undirected graph has a closed Euler tour if and only if it is connected and each vertex has an even degree.

An undirected graph has an open Euler tour (Euler path) if it is connected, and each vertex, except for exactly two vertices, has an even degree. The two vertices of odd degree have to be 
the endpoints of the tour.

A directed graph has a closed Euler tour if and only if it is strongly connected and the in-degree of each vertex is equal to its out-degree.

Similarly, a directed graph has an open Euler tour (Euler path) if and only if for each vertex the difference between its in-degree and out-degree is 0, except for two vertices, 
where one has difference +1 (the start of the tour) and the other has difference -1 (the end of the tour) and, if you add an edge from the end to the start, the graph is strongly connected.

Definition 13.1.1.  A walk is closed if it begins and ends with the same vertex.
A trail is a walk in which no two vertices appear consecutively (in either order) more than once. (That is, no edge is used more than once.)

A tour is a closed trail.

An Euler trail is a trail in which every pair of adjacent vertices appear consecutively. (That is, every edge is used exactly once.)

An Euler tour is a closed Euler trail.

## EULER TOUR FOR SUBTREE QUERIES

Euler tour technique for subtree queries for a tree with root node 1
Note this is 1-indexed, that is the nodes are numbered from 1 to n

```py
class EulerTour:
    def __init__(self, num_nodes: int, edges: List[List[int]]):
        self.num_nodes = num_nodes
        self.edges = edges
        self.adj_list = [[] for _ in range(num_nodes + 1)]
        self.root_node = 1 # root of the tree
        self.enter_counter, self.exit_counter = [0]*(num_nodes + 1), [0]*(num_nodes + 1)
        self.counter = 1
        self.build_adj_list() # adjacency list representation of the tree
        self.euler_tour(self.root_node, -1)
    
    def build_adj_list(self) -> None:
        for u, v in self.edges:
            self.adj_list[u].append(v)
            self.adj_list[v].append(u)

    def euler_tour(self, node: int, parent_node: int) -> None:
        self.enter_counter[node] = self.counter
        self.counter += 1
        for child_node in self.adj_list[node]:
            if child_node != parent_node:
                self.euler_tour(child_node, node)
        self.exit_counter[node] = self.counter - 1
```

Implemented in C++

```cpp
class EulerTour {
public:
    int num_nodes;
    vector<vector<int>> edges;
    vector<vector<int>> adj_list;
    int root_node;
    vector<int> enter_counter, exit_counter;
    int counter;

    EulerTour(int n, vector<vector<int>>& e) {
        num_nodes = n;
        edges = e;
        adj_list.resize(num_nodes + 1);
        root_node = 1;
        enter_counter.resize(num_nodes + 1);
        exit_counter.resize(num_nodes + 1);
        counter = 1;
        build_adj_list();
        euler_tour(root_node, -1);
    }

    void build_adj_list() {
        for (auto edge : edges) {
            int u = edge[0], v = edge[1];
            adj_list[u].push_back(v);
            adj_list[v].push_back(u);
        }
    }

    void euler_tour(int node, int parent_node) {
        enter_counter[node] = counter;
        counter++;
        for (auto child_node : adj_list[node]) {
            if (child_node != parent_node) {
                euler_tour(child_node, node);
            }
        }
        exit_counter[node] = counter - 1;
    }
};
```

## EULER TOUR FOR PATH QUERIES 

This one always increments the counter so that enter and exit counter will be differeent for each node. 

Allows to undo operation and get the sum along a path from root to a node in O(logn) time

Uses a fenwick tree to compute the sum along a path, from root you just do fenwick_tree.query(enter_counter[node]) get's sum from root to node.

This is 1-indexed, that is the nodes are numbered from 1 to n

Example of how need to update fenwick tree for each enter/exit counter for a node that is being updated, wich a delta value (change in value from current value in array)
fenwick_tree.update(enter_counter, delta) # update the fenwick tree
fenwick_tree.update(exit_counter, -delta)


```py
class EulerTourPathQueries:
    def __init__(self, num_nodes: int, edges: List[List[int]]):
        self.num_nodes = num_nodes
        self.edges = edges
        self.adj_list = [[] for _ in range(num_nodes + 1)]
        self.root_node = 1 # root of the tree
        self.enter_counter, self.exit_counter = [0]*(num_nodes + 1), [0]*(num_nodes + 1)
        self.counter = 1
        self.build_adj_list() # adjacency list representation of the tree
        self.euler_tour(self.root_node, -1)
    
    def build_adj_list(self) -> None:
        for u, v in self.edges:
            self.adj_list[u].append(v)
            self.adj_list[v].append(u)

    def euler_tour(self, node: int, parent_node: int):
        self.enter_counter[node] = self.counter
        self.counter += 1
        for child_node in self.adj_list[node]:
            if child_node != parent_node:
                self.euler_tour(child_node, node)
        self.counter += 1
        self.exit_counter[node] = self.counter
```