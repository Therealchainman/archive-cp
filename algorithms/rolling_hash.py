"""
This works for when you have only lowercase english letters, 
This is an implementation of a rolling hash to find pattern in 
string
"""

def find_pattern_rolling_hash(string: str, pattern: str) -> int:
    p, MOD = 31, int(1e9)+7
    coefficient = lambda x: ord(x) - ord('a') + 1
    pat_hash = 0
    for ch in pattern:
        pat_hash = ((pat_hash*p)%MOD + coefficient(ch))%MOD
    POW = 1
    for _ in range(len(pattern)-1):
        POW = (POW*p)%MOD
    cur_hash = 0
    for i, ch in enumerate(string):
        cur_hash = ((cur_hash*p)%MOD + coefficient(ch))%MOD
        if i>=len(pattern)-1:
            if cur_hash == pat_hash: return i-len(pattern)+1
            cur_hash = (cur_hash - (POW*coefficient(string[i-len(pattern)+1]))%MOD + MOD)%MOD
    return -1