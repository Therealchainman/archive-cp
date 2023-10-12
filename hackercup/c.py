import sys

# name = "replay_value_input.txt"
name = "replay_value_sample_input.txt"

sys.stdout = open(f"outputs/{name}", "w")
sys.stdin = open(f"inputs/{name}", "r")

from bisect import bisect_left

def main():
    mod = int(1e9) + 7
    N, S, E = map(int, input().split())
    lasers = [None] * N
    for i in range(N):
        x, y = map(int, input().split())
        lasers[i] = [x, y]
    # y value coordinate compression
    lasers.sort(key=lambda x: x[1])
    y_values = [y for _, y in lasers]
    si, ei = bisect_left(y_values, S), bisect_left(y_values, E)
    offset = 0
    for i in range(N + 1):
        if i == si:
            S = i + offset
            offset += 1
        if i == ei:
            E = i + offset
            offset += 1
        if i < N:
            lasers[i][1] = i + offset
    # flip all the y coordinate values if E > S
    if E > S:
        for i in range(N):
            lasers[i] = (lasers[i][0], N + 1 - lasers[i][1])
        S, E = N + 1 - S, N + 1 - E
    dp = [[[[0] * (N + 2) for _ in range(N + 2)] for _ in range(N + 2)] for _ in range(N + 2)]
    dp[N + 1][0][N + 1][0] = 1
    # lasers are sorted in increasing x coordinate value
    for _, y in sorted(lasers):
        ndp = [[[[0] * (N + 2) for _ in range(N + 2)] for _ in range(N + 2)] for _ in range(N + 2)]
        for a in range(N + 2):
            for b in range(N + 2):
                for c in range(N + 2):
                    for d in range(N + 2):
                        v = dp[a][b][c][d]
                        if v == 0: continue
                        if y > S: region = 1
                        elif y > E: region = 2
                        else: region = 3
                        # right
                        if region == 3:
                            ndp[a][b][c][max(d, y)] = (ndp[a][b][c][max(d, y)] + v) % mod
                        else:
                            ndp[a][b][min(c, y)][d] = (ndp[a][b][min(c, y)][d] + v) % mod
                        # up
                        if y >= d:
                            ndp[min(a, y)][b][c][d] = (ndp[min(a, y)][b][c][d] + v) % mod
                        # left
                        if (region == 1 and y >= b) or (region in (2, 3) and y <= a):
                            ndp[a][b][c][d] = (ndp[a][b][c][d] + v) % mod
                        # down
                        if y <= c:
                            ndp[a][max(b, y)][c][d] = (ndp[a][max(b, y)][c][d] + v) % mod
        dp = ndp
    num_configs = 0
    for a in range(N + 2):
        for b in range(N + 2):
            for c in range(N + 2):
                for d in range(N + 2):
                    num_configs = (num_configs + dp[a][b][c][d]) % mod
    res = (pow(4, N, mod) - num_configs) % mod
    return res

if __name__ == '__main__':
    T = int(input())
    for t in range(1, T + 1):
        print(f"Case #{t}: {main()}")