# Tree Algorithms

## Notes

if the implementation is in python it will have this at the top of the python script for fast IO operations

```py
import os,sys
from io import BytesIO, IOBase
from typing import *

# Fast IO Region
BUFSIZE = 8192
class FastIO(IOBase):
    newlines = 0
    def __init__(self, file):
        self._fd = file.fileno()
        self.buffer = BytesIO()
        self.writable = "x" in file.mode or "r" not in file.mode
        self.write = self.buffer.write if self.writable else None
    def read(self):
        while True:
            b = os.read(self._fd, max(os.fstat(self._fd).st_size, BUFSIZE))
            if not b:
                break
            ptr = self.buffer.tell()
            self.buffer.seek(0, 2), self.buffer.write(b), self.buffer.seek(ptr)
        self.newlines = 0
        return self.buffer.read()
    def readline(self):
        while self.newlines == 0:
            b = os.read(self._fd, max(os.fstat(self._fd).st_size, BUFSIZE))
            self.newlines = b.count(b"\n") + (not b)
            ptr = self.buffer.tell()
            self.buffer.seek(0, 2), self.buffer.write(b), self.buffer.seek(ptr)
        self.newlines -= 1
        return self.buffer.readline()
    def flush(self):
        if self.writable:
            os.write(self._fd, self.buffer.getvalue())
            self.buffer.truncate(0), self.buffer.seek(0)
class IOWrapper(IOBase):
    def __init__(self, file):
        self.buffer = FastIO(file)
        self.flush = self.buffer.flush
        self.writable = self.buffer.writable
        self.write = lambda s: self.buffer.write(s.encode("ascii"))
        self.read = lambda: self.buffer.read().decode("ascii")
        self.readline = lambda: self.buffer.readline().decode("ascii")
sys.stdin, sys.stdout = IOWrapper(sys.stdin), IOWrapper(sys.stdout)
input = lambda: sys.stdin.readline().rstrip("\r\n")
```

## Solutions

## 

### Solution 1:

```py

```

## 

### Solution 1:

```py

```

## Tree Diameter

### Solution 1:  tree dp + store maximum length to leaf fro mnode + find the maximum length including the node and two paths to leaf nodes + O(n) time

This times out in python, must be implemented in c++

```py
sys.setrecursionlimit(1_000_000)

def main():
    n = int(input())
    adj_list = [[] for _ in range(n)]
    for _ in range(n - 1):
        u, v = map(int, input().split())
        u -= 1
        v -= 1
        adj_list[u].append(v)
        adj_list[v].append(u)
    max_len = 0
    leaf_lens = [0]*n
    def dfs(node: int, parent: int) -> int:
        nonlocal max_len
        if leaf_lens[node] != 0: return leaf_lens[node]
        max_leaf_len = max_leaf_len2 = 0
        for child in adj_list[node]:
            if child == parent: continue
            leaf_len = dfs(child, node)
            if leaf_len > max_leaf_len:
                max_leaf_len2 = max_leaf_len
                max_leaf_len = leaf_len
            elif leaf_len > max_leaf_len2:
                max_leaf_len2 = leaf_len
        # because I'm counting 1 for the leaf node, but really want to count from 0 at leaf and 1 at node above. but that's more complicated
        max_len = max(max_len, max_leaf_len + max_leaf_len2)
        leaf_lens[node] = max_leaf_len + 1
        return leaf_lens[node]
    dfs(0, -1)
    return max_len

if __name__ == '__main__':
    print(main())
```

```cpp
#include <bits/stdc++.h>
using namespace std;

int max_len = 0;

/*
There are n-1 edges in a tree.  

The diameter of a tree is the longest path between two nodes in the tree, in this problem the answer is
the number of edges between the two farthest nodes. 
*/
int dfs(int node, int parent, vector<vector<int>>& graph, vector<int>& leaf_lens) {
	if (leaf_lens[node] != 0) return leaf_lens[node];
	int max_leaf_len = 0, second_max_leaf_len = 0;
    for (int& nei : graph[node]) {
		if (nei == parent) continue;
		int leaf_len = dfs(nei, node, graph, leaf_lens);
		if (leaf_len > max_leaf_len) {
			second_max_leaf_len = max_leaf_len;
			max_leaf_len = leaf_len;
		} else if (leaf_len > second_max_leaf_len) {
			second_max_leaf_len = leaf_len;
		}
    }
	max_len = max(max_len, max_leaf_len + second_max_leaf_len);
	return leaf_lens[node] = max_leaf_len + 1;
}
int main() {
    int n, a,b;
    // freopen("input.txt", "r", stdin);
    // freopen("output.txt", "w", stdout);
    cin>>n;
    vector<vector<int>> graph(n+1);
    for(int i=0;i<n-1;i++){
        cin>>a>>b;
        graph[a].push_back(b);
        graph[b].push_back(a);
    }
	vector<int> leaf_lens(n+1, 0);
	dfs(1, 0, graph, leaf_lens);
	cout << max_len << endl;
}
```

### Solution 2:  given arbitrary root node find node farthest away then find the node farthest away from that node that is the diameter of tree + O(n) time

```cpp
#include <bits/stdc++.h>
using namespace std;

/*
There are n-1 edges in a tree.  

The diameter of a tree is the longest path between two nodes in the tree, in this problem the answer is
the number of edges between the two farthest nodes. 
*/
void dfs(int node, vector<vector<int>>& graph, vector<vector<int>>& dist, int i) {
    for (int& nei : graph[node]) {
        if (dist[i][nei]==0){
            dist[i][nei] = dist[i][node] + 1;
            dfs(nei, graph, dist,i);
        }
    }
}
int main() {
    int n, a,b;
    // freopen("input.txt", "r", stdin);
    // freopen("output.txt", "w", stdout);
    cin>>n;
    vector<vector<int>> graph(n+1);
    for(int i=0;i<n-1;i++){
        cin>>a>>b;
        graph[a].push_back(b);
        graph[b].push_back(a);
    }
    vector<vector<int>> dist(2,vector<int>(n+1,0));
    dist[0][1]=1;
    dfs(1,graph,dist,0);
    int node = max_element(dist[0].begin(), dist[0].end()) - dist[0].begin();
    dist[1][node]= 1;
    dfs(node,graph,dist,1);
    int diameter = *max_element(dist[1].begin(), dist[1].end())-1;
    cout<<diameter<<endl;
}
```

## Tree Distances I

### Solution 1:  two dfs + dfs1 compute longest path from node to leaf + dfs2 find if long path through parent node + O(n) time

python solution TLE but cpp accepted

```py
sys.setrecursionlimit(1_000_000)

def main():
    n = int(input())
    adj_list = [[] for _ in range(n)]
    for _ in range(n - 1):
        u, v = map(int, input().split())
        u -= 1
        v -= 1
        adj_list[u].append(v)
        adj_list[v].append(u)
    leaf_lens1, leaf_lens2 = [0] * n, [0] * n
    path_node1, path_node2 = [-1] * n, [-1] * n
    def dfs1(node: int, parent: int) -> int:
        for child in adj_list[node]:
            if child == parent: continue
            leaf_len = dfs1(child, node)
            if leaf_len > leaf_lens1[node]:
                leaf_lens2[node] = leaf_lens1[node]
                path_node2[node] = path_node1[node]
                leaf_lens1[node] = leaf_len
                path_node1[node] = child
            elif leaf_len > leaf_lens2[node]:
                leaf_lens2[node] = leaf_len
                path_node2[node] = child
        return leaf_lens1[node] + 1
    dfs1(0, -1)
    parent_lens = [0] * n
    def dfs2(node: int, parent: int) -> None:
        parent_lens[node] = parent_lens[parent] + 1 if parent != -1 else 0
        if parent != -1 and node != path_node1[parent]:
            parent_lens[node] = max(parent_lens[node], leaf_lens1[parent] + 1)
        if parent != -1 and node != path_node2[parent]:
            parent_lens[node] = max(parent_lens[node], leaf_lens2[parent] + 1)
        for child in adj_list[node]:
            if child == parent: continue
            dfs2(child, node)
    dfs2(0, -1)
    res = [max(leaf, pleaf) for leaf, pleaf in zip(leaf_lens1, parent_lens)]
    print(*res)

if __name__ == '__main__':
    main()
```

```cpp
#include <bits/stdc++.h>
using namespace std;

int dfs1(int node, int parent, vector<vector<int>>& graph, vector<int>& leaf_lens1, vector<int>& path_node1, vector<int>& leaf_lens2, vector<int>& path_node2) {
	for (int& nei : graph[node]) {
		if (nei == parent) continue;
		int leaf_len = dfs1(nei, node, graph, leaf_lens1, path_node1, leaf_lens2, path_node2);
		if (leaf_len > leaf_lens1[node]) {
			leaf_lens2[node] = leaf_lens1[node];
			path_node2[node] = path_node1[node];
			leaf_lens1[node] = leaf_len;
			path_node1[node] = nei;
		} else if (leaf_len > leaf_lens2[node]) {
			leaf_lens2[node] = leaf_len;
			path_node2[node] = nei;
		}
    }
	return leaf_lens1[node] + 1;
}
void dfs2(int node, int parent, vector<vector<int>>& graph, vector<int>& leaf_lens1, vector<int>& path_node1, vector<int>& leaf_lens2, vector<int>& path_node2, vector<int>& parent_lens) {
	parent_lens[node] = parent > 0 ? parent_lens[parent] + 1 : 0;
	if (parent > 0 && node != path_node1[parent]) {
		parent_lens[node] = max(parent_lens[node], leaf_lens1[parent] + 1);
	}
	if (parent > 0 && node != path_node2[parent]) {
		parent_lens[node] = max(parent_lens[node], leaf_lens2[parent] + 1);
	}
	for (int& nei : graph[node]) {
		if (nei == parent) continue;
		dfs2(nei, node, graph, leaf_lens1, path_node1, leaf_lens2, path_node2, parent_lens);
    }
}
int main() {
    int n, a,b;
    // freopen("input.txt", "r", stdin);
    // freopen("output.txt", "w", stdout);
    cin>>n;
    vector<vector<int>> graph(n+1);
    for(int i=0;i<n-1;i++){
        cin>>a>>b;
        graph[a].push_back(b);
        graph[b].push_back(a);
    }
	vector<int> leaf_lens1(n+1, 0), leaf_lens2(n + 1, 0), parent_lens(n + 1, 0);
	vector<int> path_node1(n + 1, 0), path_node2(n + 1, 0);
	dfs1(1, 0, graph, leaf_lens1, path_node1, leaf_lens2, path_node2);
	dfs2(1, 0, graph, leaf_lens1, path_node1, leaf_lens2, path_node2, parent_lens);
	for (int i = 1; i <= n; i++) {
		cout << max(leaf_lens1[i], parent_lens[i]) << " ";
	}
	cout << endl;
}
```

## Tree Distances II

### Solution 1:

```py

```

## Company Queries I

### Solution 1:

```py

```

## Company Queries II

### Solution 1:  the lowest common boss is the lowest common ancestor in a tree + represent the relationship between bosses as a rooted tree at node 1, since 1 is the boss of everyone

```cpp
#include <bits/stdc++.h>
using namespace std;

inline int read()
{
	int x = 0, y = 1; char c = getchar();
	while (c < '0' || c > '9') {
		if (c == '-') y = -1;
		c = getchar();
	}
	while (c >= '0' && c <= '9') x = x * 10 + c - '0', c = getchar();
	return x * y;
}

inline long long readll() {
	long long x = 0, y = 1; char c = getchar();
	while (c < '0' || c > '9') {
		if (c == '-') y = -1;
		c = getchar();
	}
	while (c >= '0' && c <= '9') x = x * 10 + c - '0', c = getchar();
	return x * y;
}

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

int main() {
    int n = read(), q = read();
    vector<vector<int>> adj_list(n);
    for (int i = 1; i < n; i++) {
        int u = read();
        u--;
        adj_list[u].push_back(i);
        adj_list[i].push_back(u);
    }
    BinaryLift binary_lift(n, adj_list);
    for (int i = 0; i < q; i++) {
        int u = read(), v = read();
        u--;
        v--;
        int lca = binary_lift.find_lca(u, v);
        cout << lca + 1 << endl;
    }
    return 0;
}
```

## Distance Queries

### Solution 1:  find lca with binary lifting to find the distance between nodes

```py
sys.setrecursionlimit(1_000_000)
import math

"""
The root node is assumed be 0
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

def main():
    n, q = map(int, input().split())
    adj_list = [[] for _ in range(n)]
    for _ in range(n - 1):
        u, v = map(int, input().split())
        u -= 1
        v -= 1
        adj_list[u].append(v)
        adj_list[v].append(u)
    binary_lift = BinaryLift(n, adj_list)
    result = []
    for _ in range(q):
        u, v = map(int, input().split())
        u -= 1
        v -= 1
        distance = binary_lift.distance(u, v)
        result.append(distance)
    return '\n'.join(map(str, result))

if __name__ == '__main__':
    print(main())
```

```cpp
#include <bits/stdc++.h>
using namespace std;

inline int read()
{
	int x = 0, y = 1; char c = getchar();
	while (c < '0' || c > '9') {
		if (c == '-') y = -1;
		c = getchar();
	}
	while (c >= '0' && c <= '9') x = x * 10 + c - '0', c = getchar();
	return x * y;
}

inline long long readll() {
	long long x = 0, y = 1; char c = getchar();
	while (c < '0' || c > '9') {
		if (c == '-') y = -1;
		c = getchar();
	}
	while (c >= '0' && c <= '9') x = x * 10 + c - '0', c = getchar();
	return x * y;
}

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

int main() {
    int n = read(), q = read();
    vector<vector<int>> adj_list(n);
    for (int i = 0; i < n-1; i++) {
        int u = read(), v = read();
        u--;
        v--;
        adj_list[u].push_back(v);
        adj_list[v].push_back(u);
    }
    BinaryLift binary_lift(n, adj_list);

    for (int i = 0; i < q; i++) {
        int u = read(), v = read();
        u--;
        v--;
        int distance = binary_lift.distance(u, v);
        cout << distance << endl;
    }
    return 0;
}
```

## Counting Paths

### Solution 1:  lowest common ancestor with binary lifting + sparse table + difference array + subtree sum with recursive dfs

```py
sys.setrecursionlimit(1_000_000)
import math

"""
The root node is assumed be 0
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
        

def main():
    n, m = map(int, input().split())
    adj_list = [[] for _ in range(n)]
    for _ in range(n - 1):
        u, v = map(int, input().split())
        u -= 1
        v -= 1
        adj_list[u].append(v)
        adj_list[v].append(u)
    binary_lift = BinaryLift(n, adj_list)
    diff_arr = [0]*n
    lca_count = [0]*n
    for _ in range(m):
        u, v = map(int, input().split())
        u -= 1
        v -= 1
        lca = binary_lift.find_lca(u, v)
        diff_arr[u] += 1
        diff_arr[v] += 1
        diff_arr[lca] -= 2
        lca_count[lca] += 1
    subtree_sum = [0]*n
    def dfs(node: int, parent_node: int) -> int:
        subtree_sum[node] = diff_arr[node]
        for nei_node in adj_list[node]:
            if nei_node != parent_node:
                subtree_sum[node] += dfs(nei_node, node)
        return subtree_sum[node]
    dfs(0, -1)
    return ' '.join(map(str, (x + y for x, y in zip(subtree_sum, lca_count))))

if __name__ == '__main__':
    print(main())
```

```cpp
#include <bits/stdc++.h>
using namespace std;

inline int read()
{
	int x = 0, y = 1; char c = getchar();
	while (c < '0' || c > '9') {
		if (c == '-') y = -1;
		c = getchar();
	}
	while (c >= '0' && c <= '9') x = x * 10 + c - '0', c = getchar();
	return x * y;
}

inline long long readll() {
	long long x = 0, y = 1; char c = getchar();
	while (c < '0' || c > '9') {
		if (c == '-') y = -1;
		c = getchar();
	}
	while (c >= '0' && c <= '9') x = x * 10 + c - '0', c = getchar();
	return x * y;
}

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

vector<int> subtree_sum;

int dfs(int node, int parent_node, vector<int>& diff_arr, vector<vector<int>>& adj_list) {
    subtree_sum[node] = diff_arr[node];
    for (int nei_node : adj_list[node]) {
        if (nei_node != parent_node) {
            subtree_sum[node] += dfs(nei_node, node, diff_arr, adj_list);
        }
    }
    return subtree_sum[node];
}

int main() {
    int n = read(), m = read();
    vector<vector<int>> adj_list(n);
    for (int i = 0; i < n-1; i++) {
        int u = read(), v = read();
        u--;
        v--;
        adj_list[u].push_back(v);
        adj_list[v].push_back(u);
    }
    BinaryLift binary_lift(n, adj_list);
    vector<int> diff_arr(n, 0);
    vector<int> lca_count(n, 0);
    for (int i = 0; i < m; i++) {
        int u = read(), v = read();
        u--;
        v--;
        int lca = binary_lift.find_lca(u, v);
        diff_arr[u]++;
        diff_arr[v]++;
        diff_arr[lca] -= 2;
        lca_count[lca]++;
    }
    subtree_sum.resize(n, 0);
    dfs(0, -1, diff_arr, adj_list);
    for (int i = 0;i < n; i++) {
        cout << subtree_sum[i] + lca_count[i] << " ";
    }
}
```

## Subtree Queries

### Solution 1:  euler tour technique for subtree queries + tree + binary index tree (Fenwick tree) + flatten tree

```py
class FenwickTree:
    def __init__(self, N):
        self.sums = [0 for _ in range(N+1)]

    def update(self, i, delta):
        while i < len(self.sums):
            self.sums[i] += delta
            i += i & (-i)

    def query(self, i):
        res = 0
        while i > 0:
            res += self.sums[i]
            i -= i & (-i)
        return res

    def __repr__(self):
        return f"array: {self.sums}"
    
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

    def euler_tour(self, node: int, parent_node: int):
        self.enter_counter[node] = self.counter
        self.counter += 1
        for child_node in self.adj_list[node]:
            if child_node != parent_node:
                self.euler_tour(child_node, node)
        self.exit_counter[node] = self.counter - 1

def main():
    n, q = map(int, input().split())
    arr = [0] + list(map(int, input().split()))
    edges = []
    for _ in range(n - 1):
        u, v = map(int, input().split())
        edges.append((u, v))
    euler_tour = EulerTour(n, edges)
    fenwick_tree = FenwickTree(n + 1)
    for node, enter_counter in enumerate(euler_tour.enter_counter[1:], start = 1):
        fenwick_tree.update(enter_counter, arr[node])
    result = []
    for _ in range(q):
        query = list(map(int, input().split()))
        if query[0] == 1:
            u, x = query[1:]
            node_index_in_flatten_tree = euler_tour.enter_counter[u]
            delta = x - arr[u]
            arr[u] = x
            fenwick_tree.update(node_index_in_flatten_tree, delta) # update the fenwick tree
        else:
            s = query[1]
            subtree_sum = fenwick_tree.query(euler_tour.exit_counter[s]) - fenwick_tree.query(euler_tour.enter_counter[s] - 1)
            result.append(subtree_sum)
    return '\n'.join(map(str, result))

if __name__ == '__main__':
    print(main())
```

```cpp
#include <bits/stdc++.h>
using namespace std;

inline int read()
{
	int x = 0, y = 1; char c = getchar();
	while (c < '0' || c > '9') {
		if (c == '-') y = -1;
		c = getchar();
	}
	while (c >= '0' && c <= '9') x = x * 10 + c - '0', c = getchar();
	return x * y;
}

inline long long readll() {
	long long x = 0, y = 1; char c = getchar();
	while (c < '0' || c > '9') {
		if (c == '-') y = -1;
		c = getchar();
	}
	while (c >= '0' && c <= '9') x = x * 10 + c - '0', c = getchar();
	return x * y;
}

long long neutral = 0;
struct FenwickTree {
    vector<long long> nodes;
    
    void init(int n) {
        nodes.assign(n + 1, neutral);
    }

    void update(int idx, long long val) {
        while (idx < (int)nodes.size()) {
            nodes[idx] += val;
            idx += (idx & -idx);
        }
    }

    int query(int left, int right) {
        return query(right) - query(left);
    }

    long long query(int idx) {
        long long result = neutral;
        while (idx > 0) {
            result += nodes[idx];
            idx -= (idx & -idx);
        }
        return result;
    }
};

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

int main() {
    // freopen("input.txt", "r", stdin);
    // freopen("output.txt", "w", stdout);
    int n = read(), q = read();
    vector<int> arr(n + 1, 0);
    for (int i = 1; i <= n; i++) {
        arr[i] = readll();
    }
    vector<vector<int>> edges;
    for (int i = 0; i < n - 1; i++) {
        int u = read(), v = read();
        edges.push_back({u, v});
    }
    EulerTour euler_tour(n, edges);
    FenwickTree fenwick_tree;
    fenwick_tree.init(n + 1);
    for (int node = 1; node <= n; node++) {
        int enter_counter = euler_tour.enter_counter[node];
        fenwick_tree.update(enter_counter, arr[node]);
    }
    for (int i = 0; i < q; i++) {
        int t = read();
        if (t == 1) {
            int u = read(); long long x = readll();
            int node_index_in_flatten_tree = euler_tour.enter_counter[u];
            int delta = x - arr[u];
            arr[u] = x;
            fenwick_tree.update(node_index_in_flatten_tree, delta);
        }
        else {
            int s = read();
            long long subtree_sum = fenwick_tree.query(euler_tour.exit_counter[s]) - fenwick_tree.query(euler_tour.enter_counter[s] - 1);
            cout << subtree_sum << endl;
        }
    }
}
```

## Path Queries

### Solution 1:  euler tour for path queries + always increase counter + undo operation for the exit counter + fenwick tree

```py
sys.setrecursionlimit(1_000_000)

class FenwickTree:
    def __init__(self, N):
        self.sums = [0 for _ in range(N+1)]

    def update(self, i, delta):
        while i < len(self.sums):
            self.sums[i] += delta
            i += i & (-i)

    def query(self, i):
        res = 0
        while i > 0:
            res += self.sums[i]
            i -= i & (-i)
        return res

    def __repr__(self):
        return f"array: {self.sums}"
    
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

def main():
    n, q = map(int, input().split())
    arr = [0] + list(map(int, input().split()))
    edges = []
    for _ in range(n - 1):
        u, v = map(int, input().split())
        edges.append((u, v))
    euler_tour = EulerTourPathQueries(n, edges)
    fenwick_tree = FenwickTree(2*n + 2)
    for node, (enter_counter, exit_counter) in enumerate(zip(euler_tour.enter_counter[1:], euler_tour.exit_counter[1:]), start = 1):
        fenwick_tree.update(enter_counter, arr[node])
        fenwick_tree.update(exit_counter, -arr[node])
    result = []
    for _ in range(q):
        query = list(map(int, input().split()))
        if query[0] == 1:
            u, x = query[1:]
            enter_counter, exit_counter = euler_tour.enter_counter[u], euler_tour.exit_counter[u]
            delta = x - arr[u]
            arr[u] = x
            fenwick_tree.update(enter_counter, delta) # update the fenwick tree
            fenwick_tree.update(exit_counter, -delta)
        else:
            s = query[1]
            path_sum = fenwick_tree.query(euler_tour.enter_counter[s])
            result.append(path_sum)
    return '\n'.join(map(str, result))

if __name__ == '__main__':
    print(main())
```

## Path Queries II

### Solution 1:

```py

```

## 

### Solution 1:

```py

```

## 

### Solution 1:

```py

```