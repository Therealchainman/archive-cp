import sys

# name = "two_apples_a_day_sample_input.txt"
# name = "two_apples_a_day_validation_input.txt"
# name = "here_comes_santa_claus_sample_input.txt"
# name = "here_comes_santa_claus_validation_input.txt"
# name = "here_comes_santa_claus_input.txt"
name = "here_comes_santa_claus_input.txt"

sys.stdout = open(f"outputs/{name}", "w")
sys.stdin = open(f"inputs/{name}", "r")

def main():
    N = int(input())
    arr = sorted(map(int, input().split()))
    if N == 5: # special case where 3 elves need to be working on one of the toys on the leftmost and rightmost toys.
        res = 0
        first = (arr[0] + arr[2]) / 2
        last = (arr[-2] + arr[-1]) / 2
        res = max(res, last - first)
        first = (arr[0] + arr[1]) / 2
        last = (arr[-3] + arr[-1]) / 2
        res = max(res, last - first)
        return res
    first = (arr[0] + arr[1]) / 2
    last = (arr[-2] + arr[-1]) / 2
    res = last - first
    return res

if __name__ == '__main__':
    T = int(input())
    for t in range(1, T + 1):
        print(f"Case #{t}: {main()}")