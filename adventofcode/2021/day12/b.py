from collections import defaultdict
# sys.stdout = open('outputs/output.txt', 'w')
with open("inputs/input.txt", "r") as f:
    raw_data = [(x.split('-')[0], x.split('-')[1]) for x in f.read().splitlines()]
    graph = defaultdict(list)
    for x, y in raw_data:
        graph[x].append(y)
        graph[y].append(x)
    visited = {'start'}
    twice = False
    def dfs(node):
        global twice
        if node == 'end':
            return 1
        res = 0
        for nei in graph[node]:
            if nei not in visited:
                if nei.islower():
                    visited.add(nei)
                res += dfs(nei)
                if nei.islower():
                    visited.remove(nei)
            elif nei.islower() and nei in visited and not twice and nei != 'start':
                twice = True
                res += dfs(nei)
                twice = False
        return res
    print(dfs('start'))
# sys.stdout.close