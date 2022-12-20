from collections import deque
import time

def main():
    with open("input.txt", 'r') as f:
        decryption_key = 811589153
        data = [(i, v*decryption_key) for i, v in enumerate(map(int, f.read().splitlines()))]
        n = len(data) - 1
        dq = deque(data)
        start = None
        rounds = 10
        for _ in range(rounds):
            for i, v in data:
                if v == 0: 
                    start = (i, v)
                    continue
                index = dq.index((i, v)) # O(n)
                dq.remove((i, v)) # O(n)
                dq.insert((index+v)%n, (i, v)) # O(n)
        distances = [1000, 2000, 3000]
        res = 0
        index = dq.index(start)
        for i in distances:
            res += dq[(index+i)%(n+1)][1]
        return res

if __name__ == '__main__':
    start_time = time.perf_counter()
    print(main())
    end_time = time.perf_counter()
    print(f'Time Elapsed: {end_time - start_time} seconds')