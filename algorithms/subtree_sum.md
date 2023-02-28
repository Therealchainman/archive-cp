# SUBTREE SUM

## RECURSIVE DFS

subtree sum from a difference array

```py
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
```