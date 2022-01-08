diff = list(range(-9,10))
arr = list(range(1,10))
for n in range(2,19):
    for d in diff:
        for start in range(1,10):
            cand = ""
            if d==0:
                cand = "".join(map(str,[start]*n))
            elif d>0:
                for v in range(start,10,d):
                    cand += str(v)
                    if len(cand)==n:
                        break
            else:
                for v in range(start,-1,d):
                    cand += str(v)
                    if len(cand)==n:
                        break
            if len(cand) == n:
                arr.append(int(cand))
# print(sorted(arr))


print(int(1e17))
