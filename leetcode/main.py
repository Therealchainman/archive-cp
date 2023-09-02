class Solution:
    def canBeEqual(self, s1: str, s2: str) -> bool:
        for mask1 in range(1 << 2):
            for mask2 in range(1 << 2):
                tmp1, tmp2 = list(s1), list(s2)
                for i in range(2):
                    if (mask1 >> i) & 1:
                        tmp1[i], tmp1[i + 2] = tmp1[i + 2], tmp1[i]
                    if (mask2 >> i) & 1:
                        tmp2[i], tmp2[i + 2] = tmp2[i + 2], tmp2[i]
                if tmp1 == tmp2: return True
        return False