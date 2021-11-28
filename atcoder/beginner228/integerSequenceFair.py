
if __name__ == '__main__':
    N,K,M = map(int,input().split())
    MOD = 998244353
    if M%MOD==0:
        print(0)
        exit()
    print(pow(M,pow(K,N,MOD-1),MOD))