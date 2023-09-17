# EULERIAN 

## Eulerian Paths

### Hierholzer's Algorithm

This is example, essentially it just consists of visiting all nodes and once all nodes visited it gets added to the stack.  In this specific example it required sorting of edges as well, but that is not necessary for Hierholzer's algorithm.  It's just the dfs that is essential and the stack. 

```py
tickets.sort(key = lambda x: x[-1], reverse = True)
adj_list = defaultdict(list)
for u, v in tickets:
    adj_list[u].append(v)
stack = []
def dfs(u):
    while adj_list[u]:
        dfs(adj_list[u].pop())
    stack.append(u)
dfs("JFK")
return stack[::-1]
```