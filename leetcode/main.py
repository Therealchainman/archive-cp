class Solution:
    def maximumNumberOfStringPairs(self, words: List[str]) -> int:
        n = len(words)
        res = 0
        vis = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if vis[j]: continue
                if words[i] == words[j][::-1]:
                    vis[j] = 1
                    res += 1
                    break
        return res