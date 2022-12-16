from parse import compile
from collections import defaultdict, deque
import time
from typing import List, Dict

def bfs(src: int, dst: int, adj_list: Dict[int, List[int]]) -> int:
    dist = 0
    queue = deque([src])
    vis = set([src])
    while queue:
        sz = len(queue)
        for _ in range(sz):
            node = queue.popleft()
            if node == dst: return dist
            for nei in adj_list[node]:
                if nei in vis: continue
                queue.append(nei)
                vis.add(nei)
        dist += 1
    return -1

def main():
    with open("input.txt", 'r') as f:
        data = f.read().splitlines()
        pat1 = compile("Valve {} has flow rate={:d}; tunnels lead to valves {}")
        pat2 = compile("Valve {} has flow rate={:d}; tunnel leads to valve {}")
        adj_list = defaultdict(list)
        valve_mask = {}
        flow_rates = {}
        for line in data:
            valve, flow, neighbors = pat1.parse(line) if not isinstance(pat1.parse(line), type(None)) else pat2.parse(line)
            neighbors = map(lambda x: x.strip(','), neighbors.split())
            valve_mask[valve] = 1 << len(valve_mask) # key: valve string name, value: integer representation (2^i) 
            flow_rates[valve] = flow
            adj_list[valve].extend(neighbors)
        valves_nonzero_flow = [k for k, v in flow_rates.items() if v]
        minDist = {}
        for src in ['AA'] + valves_nonzero_flow:
            for dst in valves_nonzero_flow:
                if src == dst: continue
                minDist[(src, dst)] = bfs(src, dst, adj_list)
        total_time = 26
        memo = {}
        res = 0
        states = deque([('AA', 0, 0, 0, False)]) # (valve, opened, pressure, time, elephants_turn)
        while states:
            pos, opened, pressure, time, elephants_turn = states.popleft()
            res = max(res, pressure)
            if not elephants_turn:
                states.append(('AA', opened, pressure, 0, True))
            if time == total_time: continue
            for dst in valves_nonzero_flow:
                mask = valve_mask[dst]
                if (opened & mask) != 0: continue
                dist = minDist[(pos, dst)]
                if total_time-dist-1-time < 0: continue
                next_pressure = pressure + flow_rates[dst]*(total_time-dist-1-time)
                key = (mask, opened|mask, time, elephants_turn) # O(n * 2^n * t * num_players)
                if key in memo and next_pressure <= memo[key]: continue
                memo[key] = next_pressure
                states.append((dst, opened|mask, next_pressure, time+dist+1, elephants_turn))
        return res

if __name__ == '__main__':
    start_time = time.perf_counter()
    print(main())
    end_time = time.perf_counter()
    print(f'Time Elapsed: {end_time - start_time:,} seconds')