from sys import *
setrecursionlimit(int(1e6))
settrace
# import faulthandler
# faulthandler.enable()
def main(t):
    N = int(f.readline())
    arr = list(map(int, f.readline().split()))
    # N = int(input())
    # arr = list(map(int, input().split()))
    adj_list = [[] for _ in range(N)]
    for _ in range(N-1):
        u, v = map(int, f.readline().split())
        # u, v = map(int, input().split())
        adj_list[u-1].append(v-1)
        adj_list[v-1].append(u-1)
    size = [1]*N
    def smaller(node, parent):
        if t == 42:
            print(f'node: {node}, parent: {parent}, adj: {adj_list[node]}', file = out)
        for child in adj_list[node]:
            if child == parent: continue
            child_small_segment_size = smaller(child, node)
            if arr[child] < arr[node]:
                size[node] += child_small_segment_size
        return size[node]
    smaller(0, -1)
    def larger(node, parent):
        for child in adj_list[node]:
            if child == parent: continue
            if arr[child] > arr[node]:
                size[child] += size[node]
            larger(child, node)
    larger(0, -1)
    return max(size)
if __name__ == '__main__':
    with open('test_data/test_set_2/ts2_input.txt', 'r') as f, open('output.txt', 'w') as out:
        # T = int(input())
        T = int(f.readline())
        for t in range(1, T+1):
            print(f'Case #{t}: {main(t)}')
