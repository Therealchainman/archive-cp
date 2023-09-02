# Leetcode Weekly Contest 274

## 2124. Check if All A's Appears Before All B's

### Solution: Retrun false if you find "ba" in the string

```c++
bool checkString(string s) {
    return s.find("ba")==string::npos;
}
```

```py
def checkString(self, s: str) -> bool:
    return s.find("ba")==-1
```

## 2125. Number of Laser Beams in a Bank

### Solution: count 1s in each row

```py
def numberOfBeams(self, bank: List[str]) -> int:
    counts = [x.count('1') for x in bank]
    prev = 0
    numBeams = 0
    for cnt in counts:
        if cnt>0:
            numBeams += (cnt*prev)
            prev = cnt
    return numBeams
```

## 2126. Destroying Asteroids

### Solution: sort asteroids and greedily destroy them until it is greater than total mass so far

```c++
const int INF = 1e5;
bool asteroidsDestroyed(int mass, vector<int>& asteroids) {
    int n = asteroids.size();
    sort(asteroids.begin(),asteroids.end());
    for (int asteroid : asteroids) {
        if (asteroid>mass) {return false;}
        if (mass>=INF) {return true;}
        mass+=asteroid;
    }
    return true;
}
```

## 2127. Maximum Employees to Be Invited to a Meeting

### Solution: Union Find (disjoint set) + topological sort with bfs

There are two cases:
1) The answer is the longest cycle
2) The answer is the sum of the longest acyclic path on all connected components with a 2-cycle. 
For this problem we put all of the nodes that are in a cycle in a disjoint set, this allows us to easily find the largest cycle in the functional
successor graph.  
We used a bfs for topological sort to find the length of acyclic paths, that I store in dist[i] the longest length traversed to node i.  
This way when we find 2-cycles we add them all to the sum for the dist[i]+dist[favorite[i]].

```c++
struct UnionFind {
    vector<int> parents, size;
    void init(int n) {
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

    bool uunion(int i, int j) {
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
class Solution {
public:
    int maximumInvitations(vector<int>& favorite) {
        int n = favorite.size();
        vector<int> indegrees(n,0), dist(n,1);
        for (int i = 0;i<n;i++) {
            indegrees[favorite[i]]++;
        }
        UnionFind ds;
        ds.init(n);
        queue<int> q;
        for (int i = 0;i<n;i++) {
            if (!indegrees[i]) {
                q.push(i);
            }
        }
        while (!q.empty()) {
            int node = q.front();
            q.pop();
            int v = favorite[node];
            dist[v] = max(dist[v], dist[node]+1);
            if (--indegrees[v]==0) {
                q.push(v);
            }
        }
        for (int i = 0;i<n;i++) {
            if (indegrees[i]) {
                ds.uunion(i,favorite[i]);
            }
        }
        int sum = 0, maxCycle = 0;
        for (int i = 0;i<n;i++) {
            if (!indegrees[i]) continue; // only want to consider those in cycle
            int len = ds.size[ds.find(i)];
            if (len==2) {
                indegrees[favorite[i]]--; // so doesn't start from this cycle as well. avoid double counting
                sum += dist[i]+dist[favorite[i]];
            } else {
                maxCycle = max(maxCycle, ds.size[ds.find(i)]);
            }
        }
        return max(maxCycle, sum);
    }
};
```

### Solution: DFS 
DFS for finding the longest cycle
DFS for finding the longest acyclic path attached to the 2-cycles.  

```c++

```