class Solution:
    def sumDistance(self, nums: List[int], s: str, d: int) -> int:
        pivot = None
        n = len(nums)
        items = sorted([(num, direction) for num, direction in zip(nums, s)])
        positions = [0] * n
        queue = deque()
        for i in range(n):
            if items[i][1] == 'R':
                queue.append(i)
                if len(queue) == 1:
                    pivot = items[i][0]
            elif queue:
                delta = items[i][0] - pivot
                moves = delta // 2
                while queue:
                    nums[queue.popleft()] = pivot
            else:
                positions[i]
        return 0