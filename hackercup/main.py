import sys

# name = "two_apples_a_day_sample_input.txt"
# name = "two_apples_a_day_validation_input.txt"
name = "two_apples_a_day_input.txt"

sys.stdout = open(f"outputs/{name}", "w")
sys.stdin = open(f"inputs/{name}", "r")

def main(t):
    res = 0
    print(f"Case #{t}: {res}")

if __name__ == '__main__':
    T = int(input())
    for t in range(1, T + 1):
        main(t)