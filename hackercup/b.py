import sys

name = "all_critical_input.txt"

sys.stdout = open(f"outputs/{name}", "w")
sys.stdin = open(f"inputs/{name}", "r")

def main():
    p = float(input())
    prob = [[0.0] * 21 for _ in range(21)]
    prob[0][0] = 1.0
    for i in range(1, 21): # number of trials
        for j in range(i + 1): # number of successes
            prob[i][j] = prob[i - 1][j] * (1 - p)
            if j > 0:
                prob[i][j] += prob[i - 1][j - 1] * p
    dp = [0.0] * 21
    dp[0] = 1.0
    res = 0
    play = 1
    precision_threshold = 1e-8
    while True:
        ndp = [0.0] * 21
        for i in range(21):
            for j in range(i + 1): # going from state of j successes to i successes, number of successes will be i - j
                ndp[i] += dp[j] * prob[20 - j][i - j] # why is this multiplication? think about that
        delta = ndp[20] - dp[20]
        if play * delta < precision_threshold and dp[0] < precision_threshold:
            break
        res += play * delta
        dp = ndp
        play += 1
    return f"{round(res, 5):.5f}"

if __name__ == '__main__':
    T = int(input())
    for t in range(1, T + 1):
        print(f"Case #{t}: {main()}")