import sys
problem = sys.argv[0].split('.')[0]
validation = ''
def grow(R, C):
    return ('^'*C for _ in range(R))
def main():
    R, C = map(int,f.readline().split())
    empty, tree = '.', '^'
    arr = [f.readline().rstrip() for _ in range(R)]
    if all(t==empty for row in arr for t in row):
        return 'Possible\n' + '\n'.join(arr)
    if R ==1 or C == 1:
        return 'Impossible'
    return 'Possible\n' + '\n'.join(grow(R, C))

if __name__ == '__main__':
    result = []
    with open(f'inputs/{problem}_{validation}input.txt', 'r') as f:
        T = int(f.readline())
        for t in range(1,T+1):
            result.append(f'Case #{t}: {main()}')
    with open(f'outputs/{problem}_output.txt', 'w') as f:
        f.write('\n'.join(result))