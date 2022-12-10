class Solution:
    def frequencySort(self, s: str) -> str:
        res = sum([[key*freq] for key, freq in map(lambda pair: (pair[0], list(pair[1])[0][1]), groupby(Counter(s).most_common(), key = lambda pair: pair[0]))], start = [])
        return ''.join(res)

