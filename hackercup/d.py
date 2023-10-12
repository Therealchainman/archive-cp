import sys

# name = "seafood_sample_input.txt"
name = "seafood_input.txt"

sys.stdout = open(f"outputs/{name}", "w")
sys.stdin = open(f"inputs/{name}", "r")

import math

def main():
    N = int(input())
    P1, P2, Ap, Bp, Cp, Dp = map(int, input().split())
    H1, H2, Ah, Bh, Ch, Dh = map(int, input().split())
    O = input()
    data = [None] * N
    for i in range(N):
        if i == 0:
            pos, har = P1, H1
        elif i == 1:
            pos, har = P2, H2
        else:
            pos = (Ap * data[i - 2][0] + Bp * data[i - 1][0] + Cp) % Dp + 1
            har = (Ah * data[i - 2][1] + Bh * data[i - 1][1] + Ch) % Dh + 1
        data[i] = (pos, har, O[i])
    data.sort()
    positions, hardness, O = zip(*data)
    if max((h for i, h in enumerate(hardness) if O[i] == "C"), default = 0) >= max((h for i, h in enumerate(hardness) if O[i] == "R"), default = 0): return -1
    def bsearch(target, stack):
        left, right = 0, len(stack) - 1
        while left < right:
            mid = (left + right + 1) >> 1
            if hardness[stack[mid]] > target: left = mid
            else: right = mid - 1
        return left
    last_clam = first_clam = N
    clam_hardness = rock_hardness = 0
    rstack, lstack, clams = [], [], []
    # construct left stack and find the first clam
    for i in range(N):
        if O[i] == "C":
            first_clam = i
            break
        else:
            while lstack and hardness[lstack[-1]] <= hardness[i]:
                lstack.pop()
            lstack.append(i)
    # construct the right stack and find the last clam
    for i in reversed(range(N)):
        if O[i] == "C":
            last_clam = i
            break
        else:
            while rstack and hardness[rstack[-1]] <= hardness[i]:
                rstack.pop()
            rstack.append(i)
    # construct the clams with monotonically decreasing hardness from the last clam and those that will not be taken care of from a rock before the last clam
    # that is the if a rock with greater hardness is between a clam and the last clam.
    for i in range(last_clam, first_clam - 1, -1):
        if O[i] == "C":
            if hardness[i] > clam_hardness and hardness[i] >= rock_hardness: clams.append(i)
            clam_hardness = max(clam_hardness, hardness[i])
        else:
            rock_hardness = max(rock_hardness, hardness[i])
    """
    dynamic programming over the clams
    """
    clams = clams[::-1]
    nc = len(clams)
    # nearest rock with greater hardness but to the left of the clam
    nearest = [-1] * nc
    j = 0
    for i in range(first_clam, last_clam + 1):
        if O[i] == "C":
            if j < len(clams) and i == clams[j] and lstack and hardness[lstack[0]] > hardness[clams[j]]:
                k = bsearch(hardness[clams[j]], lstack)
                nearest[j] = lstack[k]
            j += i == clams[j]
        else:
            while lstack and hardness[lstack[-1]] <= hardness[i]:
                lstack.pop()
            lstack.append(i)
    rnearest = [-1] * nc
    j = nc - 1
    for i in range(last_clam, first_clam - 1, -1):
        if O[i] == "C":
            if j >= 0 and i == clams[j] and rstack and hardness[rstack[0]] > hardness[clams[j]]:
                k = bsearch(hardness[clams[j]], rstack)
                rnearest[j] = rstack[k]
            j -= i == clams[j]
        else:
            while rstack and hardness[rstack[-1]] <= hardness[i]:
                rstack.pop()
            rstack.append(i)
    dp = [math.inf] * (nc + 1)
    dp[0] = 0
    i = 1
    for k in range(first_clam, last_clam):
        if O[k] == "C":
            if clams[i - 1] != k: continue
            cnt = 0
            for j in range(i - 1, -1, -1):
                cnt += 1
                if cnt == 60: break
                if nearest[j] == -1: continue
                n = nearest[j]
                dp[i] = min(dp[i], dp[j] + 2 * (positions[clams[i - 1]] - positions[n]))
            i += 1
    # interest = []
    # for i in range(1, len(values)):
    #     interest.append(values[i] - values[i - 1])
    # print(interest)
    # case when moving to the left
    # ****L***R***L****
    # ------------>
    #         <----
    # case when moving to the right
    # *****L*****L*****R
    # ----------------->
    for j in range(nc - 1, -1, -1):
        if nearest[j] != -1:
            dp[-1] = min(dp[-1], dp[j] + 2 * positions[clams[-1]] - positions[nearest[j]])
        if rnearest[j] != -1:
            dp[-1] = min(dp[-1], dp[j] + positions[rnearest[j]])
    return dp[-1]

if __name__ == '__main__':
    T = int(input())
    for t in range(1, T + 1):
        print(f"Case #{t}: {main()}")