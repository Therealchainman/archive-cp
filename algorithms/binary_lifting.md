# BINARY LIFTING


## BINARY LIFTING TO FIND THE LCA FOR A TREE REPRESENTED BY AN ADJACENCY LIST

```py
sys.setrecursionlimit(1_000_000)
import math

"""
The root node is assumed be 0, for - indexed nodes
"""

class BinaryLift:
    """
    This binary lift function works on any undirected graph that is composed of
    an adjacency list defined by graph
    """
    def __init__(self, node_count: int, graph: List[List[int]]):
        self.size = node_count
        self.graph = graph # pass in an adjacency list to represent the graph
        self.depth = [0]*node_count
        self.parents = [-1]*node_count
        self.visited = [False]*node_count
        # ITERATE THROUGH EACH POSSIBLE TREE
        for node in range(node_count):
            if self.visited[node]: continue
            self.visited[node] = True
            self.get_parent_depth(node)
        self.maxAncestor = 18 # set it so that only up to 2^18th ancestor can exist for this example
        self.jump = [[-1]*self.maxAncestor for _ in range(self.size)]
        self.build_sparse_table()
        
    def build_sparse_table(self) -> None:
        """
        builds the jump sparse arrays for computing the 2^jth ancestor of ith node in any given query
        """
        for j in range(self.maxAncestor):
            for i in range(self.size):
                if j == 0:
                    self.jump[i][j] = self.parents[i]
                elif self.jump[i][j-1] != -1:
                    prev_ancestor = self.jump[i][j-1]
                    self.jump[i][j] = self.jump[prev_ancestor][j-1]
                    
    def get_parent_depth(self, node: int, parent_node: int = -1, depth: int = 0) -> None:
        """
        Fills out the depth array for each node and the parent array for each node
        """
        self.parents[node] = parent_node
        self.depth[node] = depth
        for nei_node in self.graph[node]:
            if self.visited[nei_node]: continue
            self.visited[nei_node] = True
            self.get_parent_depth(nei_node, node, depth+1)

    def distance(self, p: int, q: int) -> int:
        """
        Computes the distance between two nodes
        """
        lca = self.find_lca(p, q)
        return self.depth[p] + self.depth[q] - 2*self.depth[lca]

    def find_lca(self, p: int, q: int) -> int:
        # ASSUME NODE P IS DEEPER THAN NODE Q   
        if self.depth[p] < self.depth[q]:
            p, q = q, p
        # PUT ON SAME DEPTH BY FINDING THE KTH ANCESTOR
        k = self.depth[p] - self.depth[q]
        p = self.kthAncestor(p, k)
        if p == q: return p
        for j in range(self.maxAncestor)[::-1]:
            if self.jump[p][j] != self.jump[q][j]:
                p, q = self.jump[p][j], self.jump[q][j] # jump to 2^jth ancestor nodes
        return self.jump[p][0]
    
    def kthAncestor(self, node: int, k: int) -> int:
        while node != -1 and k>0:
            i = int(math.log2(k))
            node = self.jump[node][i]
            k-=(1<<i)
        return node
```

```cpp
class BinaryLift {
private:
    int size;
    vector<vector<int>> graph;
    vector<int> depth;
    vector<int> parents;
    vector<bool> visited;
    int maxAncestor;
    vector<vector<int>> jump;

    void get_parent_depth(int node, int parent_node = -1, int depth = 0) {
        parents[node] = parent_node;
        this->depth[node] = depth;
        for (int nei_node : graph[node]) {
            if (visited[nei_node]) continue;
            visited[nei_node] = true;
            get_parent_depth(nei_node, node, depth+1);
        }
    }

    void build_sparse_table() {
        for (int j = 0; j < maxAncestor; j++) {
            for (int i = 0; i < size; i++) {
                if (j == 0) {
                    jump[i][j] = parents[i];
                } else if (jump[i][j-1] != -1) {
                    int prev_ancestor = jump[i][j-1];
                    jump[i][j] = jump[prev_ancestor][j-1];
                }
            }
        }
    }

public:
    BinaryLift(int node_count, vector<vector<int>>& graph) {
        size = node_count;
        this->graph = graph;
        depth.resize(node_count);
        parents.resize(node_count);
        visited.resize(node_count, false);
        for (int node = 0; node < node_count; node++) {
            if (visited[node]) continue;
            visited[node] = true;
            get_parent_depth(node);
        }
        maxAncestor = 18;
        jump.resize(size, vector<int>(maxAncestor, -1));
        build_sparse_table();
    }

    int distance(int p, int q) {
        int lca = find_lca(p, q);
        return depth[p] + depth[q] - 2 * depth[lca];
    }

    int find_lca(int p, int q) {
        if (depth[p] < depth[q]) {
            swap(p, q);
        }
        int k = depth[p] - depth[q];
        p = kthAncestor(p, k);
        if (p == q) return p;
        for (int j = maxAncestor-1; j >= 0; j--) {
            if (jump[p][j] != jump[q][j]) {
                p = jump[p][j];
                q = jump[q][j];
            }
        }
        return jump[p][0];
    }

    int kthAncestor(int node, int k) {
        while (node != -1 && k > 0) {
            int i = log2(k);
            node = jump[node][i];
            k -= (1<<i);
        }
        return node;
    }
};

```