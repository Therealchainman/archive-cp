from collections import defaultdict
from heapq import heappop, heappush
def main():
    d, n, x = map(int, f.readline().split())
    print('day, n, x', d, n, x)
    # Seed = make_dataclass('Seed', [('quantity', int), ('cost', int), ('value', int)])
    seeds = []
    for _ in range(n):
        q, l, v = map(int,f.readline().split())
        seeds.append((q,l,v))
    deadlines = defaultdict(list)
    days = set()
    for i,(_,l,_) in enumerate(seeds):
        deadlines[d-l].append(i)
        days.add(d-l)
    days = sorted(list(days),reverse=True)
    maxheap = []
    day = d
    profit = 0
    day_amount = 0
    total = 0
    for i, day in enumerate(sorted(deadlines.keys(), reverse=True)):
        indices = deadlines[day]
        for index in indices:
            q, l, v = seeds[index]
            total += q*v
            heappush(maxheap, (-v,q))
        current_day = day
        next_day = 0
        f1.write(f'maxheap: {maxheap}\n')
        if i < len(days)-1:
            next_day = days[i+1]
        # print(next_day, current_day, maxheap)
        while maxheap and current_day > next_day:
            v, q = heappop(maxheap)
            v = abs(v)
            if day_amount > 0:
                take = min(q, day_amount)
                q -= take
                current_day -= take//day_amount
                day_amount -= take
                profit += take*v
            else:
                take = min(q, (current_day-next_day)*x)
                profit += take*v
                current_day -= take//x
                day_amount = take%x
                q -= take
            if q > 0:
                heappush(maxheap, (-v, q))
        f1.write(f'profit: {profit}\n')
        f1.write(f'current_day: {current_day}\n')
    print(maxheap)
    print(total)
    return profit

    
if __name__ == '__main__':
    result = []
    with open('logs.txt', 'w') as f1:
        with open(f'/home/therealchainman/cp/archive-cp/kickstart/2022/inputs/input.txt', 'r') as f:
            T = int(f.readline())
            for t in range(1,T+1):
                result.append(f'Case #{t}: {main()}')
    with open('/home/therealchainman/cp/archive-cp/kickstart/2022/outputs/output.txt', 'w') as f:
        f.write('\n'.join(result))