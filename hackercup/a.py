import sys

name = "ethan_finds_the_shortest_path_input.txt"

sys.stdout = open(f"outputs/{name}", "w")
sys.stdin = open(f"inputs/{name}", "r")

def main(t):
    N, K = map(int, input().split())
    edges = [(1, N, K)]
    s = 0
    u = 1
    if N > 2 and K > 2:
        for i in range(K - 1, 0, -1):
            if u == N: break
            if i == 1: edges.append((u, N, i))
            else: edges.append((u, u + 1, i))
            u += 1
            s += i
    res = max(0, s - K)
    print(f"Case #{t}: {res}")
    print(len(edges))
    for edge in edges:
        print(*edge)

if __name__ == '__main__':
    T = int(input())
    for t in range(1, T + 1):
        main(t)
        # print(f"Case #{t}: {main()}")