import heapq
with open("inputs/input.txt", "r") as f:
    data = []
    lines = f.read().splitlines()
    for line in lines:
        data.append([int(x) for x in line])
    heap = []
    R, C = len(data), len(data[0])
    heapq.heappush(heap,(0,0,0))
    vis = set()
    vis.add((0,0))
    while heap:
        cost, r, c = heapq.heappop(heap)
        if r==5*R-1 and c==5*C-1:
            print(cost)
            break
        for dr in range(-1,2):
            for dc in range(-1,2):
                if abs(dr+dc)==1:
                    nr, nc = r+dr, c+dc
                    if 0<=nr<5*R and 0<=nc<5*C and (nr,nc) not in vis:
                        x, y = nr//R, nc//C
                        nval = data[nr-x*R][nc-y*C] + x + y
                        nval = nval if nval<=9 else nval-9
                        vis.add((nr,nc))
                        heapq.heappush(heap,(cost+nval,nr,nc))