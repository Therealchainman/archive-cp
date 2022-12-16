from parse import compile
from collections import defaultdict

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
            flow_rates[valve_mask[valve]] = flow
            adj_list[valve_mask[valve]].extend(neighbors)
        total_time = 30
        memo = {}
        states = [(valve_mask['AA'], 0, 0)] # (valve_num, opened_mask, pressure)
        for t in range(1, total_time+1):
            new_states = []
            for valve_num, opened_mask, pressure in states:
                key = (valve_num, opened_mask)
                if key in memo and pressure <= memo[key]:
                    continue
                memo[key] = pressure
                flow_rate = flow_rates[valve_num]
                if valve_num & opened_mask == 0 and flow_rate > 0:
                    new_states.append((valve_num, opened_mask | valve_num, pressure + flow_rate * (total_time - t)))
                for nei_node in adj_list[valve_num]:
                    new_states.append((valve_mask[nei_node], opened_mask, pressure))
            states = new_states
        return max(pressure for _, _, pressure in states)

if __name__ == '__main__':
    print(main())