class Solution:
    def maximumDetonation(self, bombs: List[List[int]]) -> int:
        euclidean_dist = lambda p1, p2: (p2[0] - p1[0])**2 + (p2[1] - p1[1])**2
        n = len(bombs)
        adj_list = [[] for _ in range(n)]
        for i in range(n):
            x1, y1, r1 = bombs[i]
            for j in range(n):
                if i == j: continue
                x2, y2, r2 = bombs[j]
                if euclidean_dist((x1, y1), (x2, y2)) <= r1 * r1:
                    adj_list[i].append(j)
        res = 0
        for start_bomb in range(n):
            visited = [0] * n
            visited[start_bomb] = 1
            queue = deque([start_bomb])
            while queue:
                bomb = queue.popleft()
                for nei in adj_list[bomb]:
                    if visited[nei]: continue
                    visited[nei] = 1
                    queue.append(nei)
            res = max(res, sum(visited))
        return res
