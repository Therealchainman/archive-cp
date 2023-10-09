import sys

# name = "two_apples_a_day_sample_input.txt"
# name = "two_apples_a_day_validation_input.txt"
# name = "two_apples_a_day_input.txt"
name = "perfectly_balanced_chapter_1_input.txt"
# name = "perfectly_balanced_chapter_1_sample_input.txt"

sys.stdout = open(f"outputs/{name}", "w")
sys.stdin = open(f"inputs/{name}", "r")

def main():
    s = input()
    Q = int(input())
    n = len(s)
    unicode = lambda ch: ord(ch) - ord('a')
    psum = [[0] * 26 for _ in range(n + 1)]
    for i in range(n):
        for j in range(26):
            psum[i + 1][j] = psum[i][j]
        psum[i + 1][unicode(s[i])] += 1
    res = 0
    for i in range(Q):
        left, right = map(int, input().split())
        left -= 1
        right -= 1
        if (right - left + 1) % 2 == 0: continue
        mid = (left + right) // 2
        # part 1
        lsum, rsum = [psum[mid + 1][j] - psum[left][j] for j in range(26)], [psum[right + 1][j] - psum[mid + 1][j] for j in range(26)]
        off = sum([abs(lsum[j] - rsum[j]) for j in range(26)])
        if off == 1:
            res += 1
            continue
        # part 2
        lsum, rsum = [psum[mid][j] - psum[left][j] for j in range(26)], [psum[right + 1][j] - psum[mid][j] for j in range(26)]
        off = sum([abs(lsum[j] - rsum[j]) for j in range(26)])
        if off == 1:
            res += 1
    return res

if __name__ == '__main__':
    T = int(input())
    for t in range(1, T + 1):
        print(f"Case #{t}: {main()}")