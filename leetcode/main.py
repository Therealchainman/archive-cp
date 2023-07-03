def is_prime(x: int) -> bool:
    if x < 2: return False
    if x == 2: return True
    if x % 2 == 0: return False
    for i in range(3, int(math.sqrt(x)) + 1, 2):
        if x % i == 0: return False
    return True

class Solution:
    def findPrimePairs(self, n: int) -> List[List[int]]:
        res = []
        if n == 4:
            res.append([2, 2])
        for x in range(3, n // 2 + 1, 2):
            y = n - x
            if not is_prime(x) or not is_prime(y): continue
            res.append([x, y])
        return res

"""

999959
999961
999773
999809
"""