from collections import defaultdict
A = defaultdict(list)
mod = 1007
for i in range(1,11):
    for j in range(i+1,11):
        for k in range(j+1,11):
            A[(i%mod)^(j%mod)^(k%mod)].append((i,j,k))
print(A)