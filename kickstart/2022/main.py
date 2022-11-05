def main():
    n = int(input())
    arr = list(map(int, input().split()))
    total = 0
    for i in range(n):
        prefix = 0
        for j in range(i, n):
            prefix += arr[j]
            if prefix < 0: break
            total += prefix
    return total
    
if __name__ == '__main__':
    T = int(input())
    for t in range(1, T+1):
        print(f'Case #{t}: {main()}')