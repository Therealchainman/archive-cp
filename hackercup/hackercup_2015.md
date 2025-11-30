# Meta Hacker Cup 2015

# Round 2

## 

```py

```

## Problem B: All Critical

binomial distribution, probability, expected value, calculate binomial distribution with either dp or combinatorics with factorials, dynamic programming

Note on the dynamic programming it is basically saying this is the probability of this many critical bars at this number of playthroughs of the song. 

On average how many times will you have to play a song to win all 20 critical bars

P(s, i) = probability that collected exactly i critical bars after s plays of the song.  So P(0, 0) = 1, i > 0 P(0, i) = 0.  We can compute P(s, i) for s > 0 and 0 <= i <= 20 recursively 

P(s, i) = P(s - 1, j) * C(20 - j, i - j) * p ^ (i - j) * (1 - p) ^ (20 - i)

binomial distribution => 
20 - j trials and i - j successes

Q(i) = P(i, 20) - P(i - 1, 20) Probability that we receive our 20th critical bar on the ith play through.  

E = sum(i * Q(i)) for i > 0, basically Q(i) drops off exponentially or so, so it will have not have infinite iterations if we only care about precision up to 5 decimal places. 

```py
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
```

## 

```py

```

##

```py

```

##

```py

```

##

```py

```

##

```py

```