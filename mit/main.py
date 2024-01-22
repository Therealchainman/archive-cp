import sys
sys.stdin = open("small.txt")
mod = int(1e9) + 7
N = int(input())
ranges = []
for _ in range(N):
    l, r = map(int, input().split())
    ranges.append((l, r))

points = [[]]
for l, r in ranges:
    npoints = []
    for v in range(l, r + 1):
        for poin in points:
            scores = poin[:]
            scores.append(v)
            npoints.append(scores)
    points = npoints
def check(arr):
    psum = rsum = 0
    for i in range(N - 1):
        psum += arr[i]
        rsum += arr[-i - 1]
        if psum == rsum: return False
    print("arr", arr)
    return True
ans = 0
for point in points:
    if check(point): ans += 1
print(ans)