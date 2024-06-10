class Solution:
    def findWinningPlayer(self, skills: List[int], k: int) -> int:
        n = len(skills)
        cnt = 0
        winner = skills[0]
        for i in range(1, n):
            if winner > skills[i]:
                cnt += 1
            else:
                winner = skills[i]
                cnt = 1
            if cnt == k: return i
        return max(range(n), key = lambda i: skills[i])