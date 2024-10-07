import sys

base = "substantial_losses"
name = base + "_sample_input.txt"
# name = base + "_validation_input.txt"
# name = base + "_input.txt"

sys.stdout = open(f"outputs/{name}", "w")
sys.stdin = open(f"inputs/{name}", "r")
M = 998244353
def main():
    W, G, L = map(int, input().split())
    v = (2 * L + 1) % M
    ans = ((W - G) * v) % M
    return ans

if __name__ == '__main__':
    T = int(input())
    for t in range(1, T + 1):
        print(f"Case #{t}: {main()}")