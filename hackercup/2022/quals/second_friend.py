from collections import Counter
import sys
problem = sys.argv[0].split('.')[0]
validation = 'validation_'
def main():
    n, k = map(int,f.readline().split())
    styles = list(map(int,f.readline().split()))
    if len(styles) > 2*k or any(cnt > 2 for cnt in Counter(styles).values()):
        return 'NO'
    return 'YES'

if __name__ == '__main__':
    result = []
    with open(f'inputs/{problem}_{validation}input.txt', 'r') as f:
        T = int(f.readline())
        for t in range(1,T+1):
            result.append(f'Case #{t}: {main()}')
    with open(f'outputs/{problem}_output.txt', 'w') as f:
        f.write('\n'.join(result))