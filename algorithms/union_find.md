# Union Find

## Union Find Algorithm

simple implementation using lists, and works if the nodes are integers labels from 0 to n or something.
This is also an iterative implementation which makes it better on memory and time.

```py
class UnionFind:
    def __init__(self, n: int):
        self.size = [1]*n
        self.parent = list(range(n))
    
    def find(self,i: int) -> int:
        while i != self.parent[i]:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i
    """
    returns true if the nodes were not union prior. 
    """
    def same(self,i: int,j: int) -> bool:
        i, j = self.find(i), self.find(j)
        if i!=j:
            if self.size[i] < self.size[j]:
                i,j=j,i
            self.parent[j] = i
            self.size[i] += self.size[j]
            return False
        return True
    @property
    def root_count(self):
        return sum(node == self.find(node) for node in range(len(self.parent)))
    def single_connected_component(self) -> bool:
        return self.size[self.find(0)] == len(self.parent)
    def is_same_connected_components(self, i: int, j: int) -> bool:
        return self.find(i) == self.find(j)
    def num_connected_components(self) -> int:
        return len(set(map(self.find, self.parent)))
    def size_(self, i):
        return self.size[self.find(i)]
    def __repr__(self) -> str:
        return f'parents: {[(i, parent) for i, parent in enumerate(self.parent)]}, sizes: {self.size}'
```

## Union Find using dictionary

This union find implementation is beneficial when you have non-integer nodes, and you want to use the nodes as keys in a dictionary.
Also it is space optimized for these type of node labels. 

Compact union find

```py
class UnionFind:
    def __init__(self):
        self.size = dict()
        self.parent = dict()
    
    def find(self,i: int) -> int:
        if i not in self.parent:
            self.size[i] = 1
            self.parent[i] = i
        while i != self.parent[i]:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self,i: int,j: int) -> bool:
        i, j = self.find(i), self.find(j)
        if i!=j:
            if self.size[i] < self.size[j]:
                i,j=j,i
            self.parent[j] = i
            self.size[i] += self.size[j]
            return True
        return False
    
    @property
    def root_count(self):
        return sum(node == self.find(node) for node in self.parent)

    def __repr__(self) -> str:
        return f'parents: {[(i, parent) for i, parent in enumerate(self.parent)]}, sizes: {self.size}'
```

## Best cpp implementation

This is my favorite cpp implementation, it is rather simple.

Intended for 0-indexed arrays

```cpp
struct UnionFind {
    vector<int> parents, size;
    UnionFind(int n) {
        parents.resize(n);
        iota(parents.begin(),parents.end(),0);
        size.assign(n,1);
    }

    int find(int i) {
        if (i==parents[i]) {
            return i;
        }
        return parents[i]=find(parents[i]);
    }

    bool same(int i, int j) {
        i = find(i), j = find(j);
        if (i!=j) {
            if (size[j]>size[i]) {
                swap(i,j);
            }
            size[i]+=size[j];
            parents[j]=i;
            return false;
        }
        return true;
    }
};

```

## Persistent Disjoint Union Set Data Structure

With the pointers and values, you can use that to backtrack to the state of the disjoint set at an earlier time.  
It allows time traveling to the past version of the data structure
The sum is just what we are calculating for this problem not too important. 

This does not have path compression, but it does have union by rank, which gives a time complexity of O(log(N))

```cpp
vector<int> values;
vector<int*> pointers;
struct UnionFind {
    vector<int> parents, size;
    void init(int n) {
        parents.resize(n);
        iota(parents.begin(),parents.end(),0);
        size.assign(n,1);
    }

    int find(int i) {
		if (i == parents[i]) return i;
		return find(parents[i]);
    }

    void unite(int i, int j) {
        i = find(i), j = find(j);
		if (i == j) return;
		if (comp[i] != comp[j]) return;
		if (size[j] > size[i]) swap(i, j);
		pointers.push_back(&sum);
		values.push_back(sum);
		sum = sum - cost(size[i]) - cost(size[j]) + cost(size[i] + size[j]);
		pointers.push_back(&parents[j]);
		values.push_back(parents[j]);
		parents[j] = i;
		pointers.push_back(&size[i]);
		values.push_back(size[i]);
		size[i] += size[j];
    }
};

// example of how it backtracks in a divide and conquer algorithm, so it goes back to the time at start of segment
int snap_time = values.size();
// unite
for (int i : events) {
    if (i < mid) dsu.unite(edges[i].u, edges[i].v);
}
vector<int> tol, tor;
for (int i : events) {
    if (i >= mid) tor.push_back(i);
    else if (dsu.find(edges[i].u) == dsu.find(edges[i].v)) tol.push_back(i);
    else tor.push_back(i);
}
calc(mid, right, tor);
// backtrack for the dsu
while (values.size() > snap_time) {
    *pointers.end()[-1] = values.end()[-1];
    values.pop_back();
    pointers.pop_back();
}
calc(left, mid, tol);
```

## Union Find for offline queries

This algorithm efficiently answers queries regarding the terrain height difference between two points on a 2D grid using a Union-Find (Disjoint Set Union, DSU) data structure. The problem is structured as follows:

Input Representation
The grid consists of R rows and C columns, with N = R × C total cells.
Each cell has a given height, stored in the heights vector.
A series of Q queries are given, each specifying two cells and their respective heights at a certain time.

Union-Find Data Structure with Query Storage

A Union-Find structure is used to track connected components in the grid.
Each query is associated with two specific cells (startCell[i] and endCell[i]) to determine the maximum height at which they become connected.
Queries are stored in queries[i] for each cell.

Processing Grid Heights in Decreasing Order

The cells are sorted in descending order by height.
As we process each cell from highest to lowest, we dynamically merge it with its neighbors if the neighbors have a height greater than or equal to the current height.

Merging with Union-Find

The merge(i, j, h) function ensures that two adjacent cells are in the same connected component if they should be merged.
During merging, if any query connects startCell[i] and endCell[i], the maxHeights[i] is updated.
Processing Queries

Key Differences
The advanced version of Union-Find maintains additional metadata (queries and maxHeights) to process queries efficiently, whereas the basic version only tracks connected components.
The advanced version processes elements in decreasing height order, ensuring that merges only happen when valid.
The query mechanism is integrated into the merge process rather than being handled separately.
The basic version is a general-purpose DSU, whereas this implementation is tailored for pathfinding problems involving terrain height.

```cpp
int R, C, N, Q;
vector<int> heights, cells, maxHeights, startCell, endCell, height1, height2;

struct UnionFind {
    vector<int> parents, size;
    vector<vector<int>> queries;
    UnionFind(int n) {
        parents.resize(n);
        iota(parents.begin(),parents.end(),0);
        size.assign(n,1);
        queries.assign(n, vector<int>());
    }

    int find(int i) {
        if (i==parents[i]) {
            return i;
        }
        return parents[i]=find(parents[i]);
    }

    void merge(int i, int j, int h) {
        i = find(i), j = find(j);
        if (i == j) return;
        if (queries[j].size() > queries[i].size()) {
            swap(i,j);
        }
        size[i]+=size[j];
        parents[j]=i;
        for (int qi : queries[j]) {
            if (find(startCell[qi]) == find(endCell[qi])) {
                maxHeights[qi] = max(maxHeights[qi], h);
            }
            queries[i].emplace_back(qi);
        }
        queries[j].clear();
    }

    void add(int i, int h) {
        if (i % C != C - 1 && heights[i + 1] >= heights[i]) {
            merge(i, i + 1, h);
        }
        if (i % C != 0 && heights[i - 1] >= heights[i]) {
            merge(i, i - 1, h);
        }
        if (i + C < N && heights[i + C] >= heights[i]) {
            merge(i, i + C, h);
        }
        if (i - C >= 0 && heights[i - C] >= heights[i]) {
            merge(i, i - C, h);
        }
    }
};

int map2Dto1D(int i, int j) {
    return i * C + j;
}
```