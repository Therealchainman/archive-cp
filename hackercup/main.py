import sys

# name = "two_apples_a_day_sample_input.txt"
# name = "two_apples_a_day_validation_input.txt"
name = "two_apples_a_day_input.txt"

sys.stdout = open(f"outputs/{name}", "w")
sys.stdin = open(f"inputs/{name}", "r")

def main():
    res = 0
    return res

if __name__ == '__main__':
    T = int(input())
    for t in range(1, T + 1):
        print(f"Case #{t}: {main()}")