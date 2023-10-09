import sys

# name = "balance_scale_sample_input.txt"
name = "balance_scale_input.txt"

sys.stdout = open(f"outputs/{name}", "w")
sys.stdin = open(f"inputs/{name}", "r")

mod = int(1e9 + 7)
TOTAL = 3_000 * 3_000

def mod_inverse(v):
    return pow(v, mod - 2, mod)

def factorials(n):
    fact, inv_fact = [1] * (n + 1), [0] * (n + 1)
    for i in range(2, n + 1):
        fact[i] = (fact[i - 1] * i) % mod
    inv_fact[-1] = mod_inverse(fact[-1])
    for i in reversed(range(n)):
        inv_fact[i] = (inv_fact[i + 1] * (i + 1)) % mod
    return fact, inv_fact

def main():
    N, K = map(int, input().split())
    cookies = [None] * N
    for i in range(N):
        c, w = map(int, input().split())
        cookies[i] = (c, w)
    C_1, W_1 = cookies[0]
    C_less = C_equal = C_greater = 0
    for i in range(1, N):
        c, w = cookies[i]
        if w < W_1:
            C_less += c
        elif w  == W_1:
            C_equal += c
        else:
            C_greater += c
    M = C_1 + C_less + C_equal + C_greater
    def nCr(n, r):
        return (fact[n] * inv_fact[r] * inv_fact[n - r]) % mod if n >= r else 0
    comb_neutral = nCr(M - C_greater, K + 1)
    comb_less = nCr(C_less, K + 1)
    comb_equal = comb_neutral - comb_less
    comb_total = nCr(M, K + 1)
    # P(AUB) = P(A|B) * P(B), B = equal, A = equal cookie from batch 1 is the tie breaker
    prob_equal = (comb_equal * mod_inverse(comb_total)) % mod
    conditional_prob = (C_1 * mod_inverse(C_1 + C_equal)) % mod
    res = (prob_equal * conditional_prob) % mod
    return res

if __name__ == '__main__':
    fact, inv_fact = factorials(TOTAL)
    T = int(input())
    for t in range(1, T + 1):
        print(f"Case #{t}: {main()}")