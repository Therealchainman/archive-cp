


```py
from math import log2
def main():
    n, q = map(int,input().split())
    LOG = int(log2(n)) + 1
    up = [[-1]*LOG for _ in range(n+1)]
    parents_arr = map(int,input().split())
    for i, par in zip(range(2,n+1), parents_arr):
        up[i][0] = par
    queries = []
    for _ in range(q):
        queries.append(list(map(int,input().split())))
    for j in range(1, LOG):
        for i in range(1,n+1):
            if up[i][j-1] == -1: continue
            up[i][j] = up[up[i][j-1]][j-1]
    result = []
    for node, k in queries:
        while node != -1 and k > 0:
            i = int(log2(k))
            node = up[node][i]
            k -= (1<<i)
        result.append(node)
    return '\n'.join(map(str,result))

if __name__ == '__main__':
    print(main())
```

```py
# Source: https://usaco.guide/general/io

from math import log2

def main():
    n, q = map(int,input().split())
    edges = list(map(int,input().split()))
    largest_k = 0
    parents_arr = [-1]*(n+1)
    for u, v in enumerate(edges, start=1):
        parents_arr[u] = v
    queries = []
    for _ in range(q):
        node, k = map(int,input().split())
        largest_k = max(largest_k, k)
        queries.append([node, k])
    result = []
    LOG = int(log2(largest_k))+1
    up = [[-1]*LOG for _ in range(n+1)]
    for i in range(1,n+1):
        up[i][0] = parents_arr[i]
    for j in range(1,LOG):
        for i in range(1,n+1):
            up[i][j] = up[up[i][j-1]][j-1]
    for node, k in queries:
        while k>0:
            i = int(log2(k))
            node = up[node][i]
            k-=(1<<i)
        result.append(node)
    return '\n'.join(map(str,result))


if __name__ == '__main__':
    print(main())
```

```py
from math import log2
from collections import deque

class LCA:
    def main(self):
        n, q = map(int,input().split())
        parents_gen = map(int, input().split())
        self.LOG = int(log2(n))+1
        self.up = [[-1]*self.LOG for _ in range(n+1)]
        graph = [[] for _ in range(n+1)] # travel from parent to child node
        for i, par in zip(range(2,n+1), parents_gen):
            self.up[i][0] = par
            graph[par].append(i)
        self.depth = [0]*(n+1)
        queue = deque([(1, 0)]) # (node, depth)
        while queue:
            node, dep = queue.popleft()
            self.depth[node] = dep
            for child in graph[node]:
                for j in range(1,self.LOG):
                    if self.up[child][j-1] == -1: break
                    self.up[child][j] = self.up[self.up[child][j-1]][j-1]
                queue.append((child, dep+1))
        result = []
        for _ in range(q):
            u, v = map(int,input().split())
            result.append(self.lca(u,v))
        return '\n'.join(map(str,result))

    def lca(self, u, v):
        # always depth[u] < depth[v], v is deeper node
        if self.depth[u] > self.depth[v]:
            u, v = v, u # swap the nodes
        k = self.depth[v] - self.depth[u]
        while k > 0:
            i = int(log2(k))
            v = self.up[v][i]
            k-=(1<<i)
        if u == v: return u
        for j in range(self.LOG)[::-1]:
            if self.up[u][j]==-1 or self.up[v][j]==-1 or self.up[u][j] == self.up[v][j]: continue
            u = self.up[u][j]
            v = self.up[v][j]
        return self.up[u][0]


if __name__ == '__main__':
    print(LCA().main())
```

```cpp
// Source: https://usaco.guide/general/io

#include <bits/stdc++.h>
using namespace std;

int LOG = 17;
vector<int> depth;
vector<vector<int>> up_vec;

int lca(int u, int v) {
	if (depth[u]>depth[v]) {
		swap(u,v);
	}
	int k = depth[v] - depth[u];
	printf("k=%d,u=%d,v=%d\n", k, u, v);
	for (int j = LOG-1;j>=0 && k>0;j--) {
		int pow2 = 1<<j;
		if (k<pow2) continue;
		v = up_vec[v][j];
		k-=pow2;
	}
	if (u==v) return u;
	for (int j = LOG-1;j>=0;j--) {
		cout<<j<<endl;
		cout<<up_vec[u][j]<<' '<<up_vec[v][j]<<endl;
		if (up_vec[u][j]==-1 || up_vec[v][j]==-1 || up_vec[u][j]==up_vec[v][j]) continue;
		u = up_vec[u][j];
		v = up_vec[v][j];
	}
	cout<<u<<endl;
	return up_vec[u][0];
}

int main() {
	int n, q, u, v;
	cin>>n>>q;
	vector<int> parents_vec;
	unordered_map<int,vector<int>> graph;
	for (int i = 1;i<n;i++) {
		cin>>u;
		cout<<u<<endl;
		parents_vec.push_back(u);
		graph[i].push_back(u);
	}
	depth.assign(n+1,0);
	up_vec.assign(n+1, vector<int>(LOG, -1));

	queue<pair<int,int>> qu;
	qu.emplace(1,0);
	while (!qu.empty()) {
		int node, dep;
		tie(node,dep) = qu.front();
		qu.pop();
		depth[node] = dep;
		for (int child : graph[node]) {
			up_vec[child][0] = parents_vec[child];
			qu.emplace(child, dep+1);
		}
	}
	for (int i = 0;i<n+1;i++) {
		for (int j = 0;j<LOG;j++) {
			printf("i=%d,j=%d,up=%d\n",i,j,up_vec[i][j]);
		}
	}
	while (q--) {
		cin>>u>>v;
		cout<<lca(u,v)<<endl;
	}
}

```