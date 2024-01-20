# Functional Graph

Graph where each vertex has outdegree = 1
A functional graph is a directed graph

## Detection of cycle

The following implementation can detect cycles in functional graphs.  And also backtrack and recover the list of nodes in the order they were visited in the cycle.  So you could compute anything related to the cycle.  You know the entire cycle.  And you can backtrack through the rest of the nodes as well for each weakly connected component. 

```py
class Solution:
    def countVisitedNodes(self, edges: List[int]) -> List[int]:
        n = len(edges)
        ans, vis = [0] * n, [0] * n
        def search(u):
            parent = {u: None}
            is_cycle = False
            while True:
                vis[u] = 1
                v = edges[u]
                if v in parent: 
                    is_cycle = True
                    break
                if vis[v]: break
                parent[v] = u
                u = v
            if is_cycle:
                crit_point = parent[edges[u]]
                cycle_path = []
                while u != crit_point:
                    cycle_path.append(u)
                    u = parent[u]
                len_ = len(cycle_path)
                for val in cycle_path:
                    ans[val] = len_
            while u is not None:
                ans[u] = ans[edges[u]] + 1
                u = parent[u]
        for i in range(n):
            if vis[i]: continue
            search(i)
        return ans
```