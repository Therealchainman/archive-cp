class Solution:
    def maxRunTime(self, n: int, batteries: List[int]) -> int:
        batteries.sort(reverse = True)
        hours = sum(batteries) - sum(batteries[:n])
        print('hours', hours)
        def possible(target):
            rem = hours
            for i in range(n):
                rem -= max(0, target - batteries[i])
                if rem < 0: return True
            return False
        i = bisect.bisect_left(range(10_000_000_000), False, lambda x: possible(x))
        return i