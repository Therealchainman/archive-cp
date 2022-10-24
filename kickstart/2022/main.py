from collections import defaultdict
from heapq import heappop, heappush
def main():
    d, n, x = map(int, f.readline().split())


    
if __name__ == '__main__':
    result = []
    with open('logs.txt', 'w') as f1:
        with open(f'/home/therealchainman/cp/archive-cp/kickstart/2022/inputs/input.txt', 'r') as f:
            T = int(f.readline())
            for t in range(1,T+1):
                result.append(f'Case #{t}: {main()}')
    with open('/home/therealchainman/cp/archive-cp/kickstart/2022/outputs/output.txt', 'w') as f:
        f.write('\n'.join(result))