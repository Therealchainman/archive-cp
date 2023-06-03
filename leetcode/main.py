class Solution:
    def numOfMinutes(self, n: int, headID: int, manager: List[int], informTime: List[int]) -> int:
        adj_list = [[] for _ in range(n)]
        for u, v in enumerate(manager):
            if v == -1: continue
            adj_list[v].append(u)
        res = 0
        queue = deque([(headID, 0)])
        while queue:
            node, time_ = queue.popleft()
            res = max(res, time_)
            for nei in adj_list[node]:
                queue.append((nei, time_ + informTime[node]))
        return res