import sys
problem = sys.argv[0].split('.')[0]
validation = ''
def main():
    N, K = map(int, f.readline().split())
    A = list(map(int, f.readline().split()))
    B = list(map(int, f.readline().split()))
    if K == 0: return "YES" if A == B else "NO"
    if N == 2: 
        if K%2==0: return "YES" if A == B else "NO"
        else: return "YES" if A != B else "NO"
    if K == 1 and A == B: return "NO"
    A += A
    for i in range(len(A)//2):
        if A[i] == B[0]:
            return "YES" if A[i:i+N] == B else "NO"
    return "NO"


if __name__ == '__main__':
    result = []
    with open(f'inputs/{problem}_{validation}input.txt', 'r') as f:
        T = int(f.readline  ())
        for t in range(1,T+1):
            result.append(f'Case #{t}: {main()}')
    with open(f'outputs/{problem}_output.txt', 'w') as f:
        f.write('\n'.join(result))