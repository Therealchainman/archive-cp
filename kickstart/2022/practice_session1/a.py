T = int(input())
for t in range(1,T+1):
    N, M = list(map(int,input().split()))
    sum_ = sum(num for num in map(int,input().split()))
    res = sum_-(sum_//M)*M
    print(f"Case #{t}: {res}")