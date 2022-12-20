from collections import defaultdict, deque
from parse import compile
import ray
from z3.z3 import *

pat = compile("Blueprint {:d}: Each ore robot costs {:d} ore. Each clay robot costs {:d} ore. Each obsidian robot costs {:d} ore and {:d} clay. Each geode robot costs {:d} ore and {:d} obsidian.")

class Blueprint:
    def __init__(self, id, oreore, clayore, obore, obclay, geore, geob):
        self.id = id
        self.oreore = oreore
        self.clayore = clayore
        self.obore = obore
        self.obclay = obclay
        self.geore = geore
        self.geob = geob
        
    def __repr__(self):
        return f'blueprint id: {self.id}, oreore: {self.oreore}, clayore: {self.clayore}, obore: {self.obore}, obclay: {self.obclay}, geore: {self.geore}, geob: {self.geob}'

@ray.remote
def bfs(blueprint: Blueprint) -> int:
    initial_state = (0, 0, 0, 0, 0, 0, 0, 1, 0) # geods count, geod robots count, obsidian robots count, clay robots count, 
    queue = deque([initial_state]) # geods count, obisidan count, clay count, ore count, geod robots count, obsidian robots count, clay robots count, ore robots count, time 
    vis = set([initial_state])
    best = 0
    prev_time = 0
    needed_ore = max(blueprint.oreore, blueprint.clayore, blueprint.obore, blueprint.geore)
    needed_clay = blueprint.obclay
    needed_obsidian = blueprint.geob
    cnt = 2
    while queue:
        geods, obsidian, clay, ore, geod_robots, obsidian_robots, clay_robots, ore_robots, time = queue.popleft()
        if time > 6: return 3
        # if time > prev_time:
        #     print(time, len(queue), best)
        #     prev_time = time
        if time == 24:
            best = max(best, geods)
            continue
        if geod_robots == 0 and (ore >= cnt*needed_ore or ore_robots >= needed_ore or clay_robots >= needed_clay): 
            continue 
        s = Solver()
        obsidian_count, clay_count, ore_count, geod_robot_count, obsidian_robot_count, clay_robot_count, ore_robot_count = Ints('obsidian_count clay_count ore_count geod_robot_count obsidian_robot_count clay_robot_count ore_robot_count')
        s.add(obsidian_count <= obsidian, clay_count <= clay, ore_count <= ore)
        s.add(ore_robot_count >= 0, clay_robot_count >= 0, obsidian_robot_count >= 0, geod_robot_count >= 0)
        s.add(blueprint.oreore*ore_robot_count + blueprint.clayore*clay_robot_count + blueprint.obore*obsidian_robot_count + blueprint.geore*geod_robot_count == ore_count)
        s.add(blueprint.obclay*obsidian_robot_count == clay_count)
        s.add(blueprint.geob*geod_robot_count == obsidian_count)
        s.add(ore_robot_count + clay_robot_count + obsidian_robot_count + geod_robot_count <= 1)
        geod_robot_req = If(And(obsidian >= blueprint.geob, ore >= blueprint.geore), geod_robot_count == 1, geod_robot_count == 0)
        s.add(geod_robot_req)
        obsidian_robot_req = If(And(Or(obsidian < blueprint.geob, ore < blueprint.geore), clay >= blueprint.obclay, ore >= blueprint.obore), obsidian_robot_count == 1, obsidian_robot_count == 0)
        s.add(obsidian_robot_req)
        
        while s.check() == sat:
            next_geods, next_obsidian, next_clay, next_ore, next_geod_robots, next_obsidian_robots, next_clay_robots, next_ore_robots, next_time = geods + geod_robots, obsidian + obsidian_robots - s.model()[obsidian_count].as_long(), clay + \
            clay_robots - s.model()[clay_count].as_long(), ore + ore_robots - s.model()[ore_count].as_long(), geod_robots + s.model()[geod_robot_count].as_long(), obsidian_robots + s.model()[obsidian_robot_count].as_long(), clay_robots + s.model()[clay_robot_count].as_long(), ore_robots + \
            s.model()[ore_robot_count].as_long(), time + 1
            # if obsidian >= blueprint.geob and ore >= blueprint.geore:
            #     print(simplify(geod_robot_req))
            #     print(next_geod_robots)
            s.add(Or(obsidian_count != s.model()[obsidian_count], clay_count != s.model()[clay_count], ore_count != s.model()[ore_count], geod_robot_count != s.model()[geod_robot_count], obsidian_robot_count != s.model()[obsidian_robot_count], clay_robot_count != s.model()[clay_robot_count], ore_robot_count != s.model()[ore_robot_count]))
            next_state = (next_geods, next_obsidian, next_clay, next_ore, next_geod_robots, next_obsidian_robots, next_clay_robots, next_ore_robots, next_time)
            if next_state not in vis:
                queue.append(next_state)
                vis.add(next_state)
    

def main():
    with open("input.txt", 'r') as f:
        data = f.read().splitlines()
        blueprints = []
        for line in data:
            blueprint, ore, clay, obore, obclay, geore, geob = pat.parse(line)
            blueprints.append(Blueprint(blueprint, ore, clay, obore, obclay, geore, geob))
        ray.init()
        results = [bfs.remote(blueprint) for blueprint in blueprints]
        results = ray.get(results)
        print(results)
        ray.shutdown()
        return sum(results)

if __name__ == '__main__':
    print(main())

