from itertools import *
class Delta:
    def __init__(self, delta: str):
        self.val = int(delta[1:]) if '+' in delta else -int(delta[1:])
def main():
    with open('input.txt', 'r') as f:
        data = map(lambda x: Delta(x), f.read().splitlines())
        seen = set()
        freq = 0
        for delta in cycle(data):
            freq += delta.val
            if freq in seen: return freq
            seen.add(freq)
        return -1
if __name__ == "__main__":
    print(main())