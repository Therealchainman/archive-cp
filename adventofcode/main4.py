from collections import defaultdict, deque
from parse import compile
import time
import ray
from functools import reduce
import operator

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

# @ray.remote
# def bfs(blueprint: Blueprint) -> int:
#     initial_state = (0, 0, 0, 0, 1, 0, 0, 0) # ore, clay, obsidian, geods, ore robots, clay robots, obsidian robots, geod robots
#     states = [initial_state] 
#     needed_ore = max(blueprint.oreore, blueprint.clayore, blueprint.obore, blueprint.geore)
#     needed_clay = blueprint.obclay
#     needed_obsidian = blueprint.geob
#     total_time = 32
#     total = []

#     for t in range(1, total_time + 1):
#         # if len(states) > 30000000: break
#         # print(t, len(states))
#         # res = 0
#         new_states = []
#         for ore, clay, obsidian, geods, ore_robots, clay_robots, obsidian_robots, geod_robots in states:
#             # print(t, ore, clay, obsidian, geods, ore_robots, clay_robots, obsidian_robots, geod_robots)
#             # pruning the tree
#             if ore >= 2*needed_ore or clay >= 6*needed_clay or ore_robots > needed_ore or clay_robots > needed_clay or obsidian_robots > needed_obsidian: continue
#             nore, nclay, nobsidian, ngeods = ore + ore_robots, clay + clay_robots, obsidian + obsidian_robots, geods + geod_robots
#             # res = max(res, nore)

#             # skip construction of robot
#             new_states.append((nore, nclay, nobsidian, ngeods, ore_robots, clay_robots, obsidian_robots, geod_robots))
#             # always construct geod robot greedily
#             if ore >= blueprint.geore and obsidian >= blueprint.geob:
#                 # print(t, ore, clay, obsidian, geods, ore_robots, clay_robots, obsidian_robots, geod_robots)
#                 new_states.append((nore - blueprint.geore, nclay, nobsidian - blueprint.geob, ngeods, ore_robots, clay_robots, obsidian_robots, geod_robots + 1))
#             elif ore >= blueprint.obore and clay >= blueprint.obclay:
#                 new_states.append((nore - blueprint.obore, nclay - blueprint.obclay, nobsidian, ngeods, ore_robots, clay_robots, obsidian_robots + 1, geod_robots))
#             else:
#                 if ore >= blueprint.oreore:
#                     new_states.append((nore - blueprint.oreore, nclay, nobsidian, ngeods, ore_robots + 1, clay_robots, obsidian_robots, geod_robots))
#                 if ore >= blueprint.clayore:
#                     new_states.append((nore - blueprint.clayore, nclay, nobsidian, ngeods, ore_robots, clay_robots + 1, obsidian_robots, geod_robots))
#         states = new_states
#         # total.append(res)
#     # print(total, len(total), len(states))
#     res = max(geods for _, _, _, geods, _, _, _, _ in states)
#     # print(res)
#     return res

@ray.remote
def get_max_geodes(blueprint, total_time):
    blueprint_id, o_o_cost, c_o_cost, ob_o_cost, ob_c_cost, g_o_cost, g_ob_cost = blueprint.id, blueprint.oreore, blueprint.clayore, blueprint.obore, blueprint.obclay, blueprint.geore, blueprint.geob
    max_o_cost = max(o_o_cost, c_o_cost, ob_o_cost, g_o_cost)

    # State is structured as:
    # (ore_robots, clay_robots, obsidian_robots, geode_robots, ore, clay, obsidian, geode, remaining time)
    initial = (1, 0, 0, 0, 0, 0, 0, 0, total_time)

    queue = deque([initial])
    seen = set()
    times = set()
    max_geodes = 0

    while queue:
        r_o, r_c, r_ob, r_g, o, c, ob, g, time = queue.popleft()

        if time not in times:
            times.add(time)
            print(f'[{blueprint_id}] Time left: {time:<2} | Queue size: {len(queue)}')

        max_geodes = max(max_geodes, g)

        if time == 0:
            continue

        # If we have more resources than we could possibly need, trim it down to reduce state space.
        # For example, note that for clay, if we have r_c clay robots and more than
        # ob_c_cost + (ob_c_cost - r_c) * (t - 1) clay, we can buy a obsidian robot next (which we are guaranteed
        # to be able to afford since we also make sure that r_c <= ob_c_cost, see below). Then for each
        # subsequent turn, if we conservatively assume we always buy another obsidian robot, after producing the next
        # batch of clay we have more than r_c + (ob_c_cost - r_c) * (t - 1) = (ob_c_cost - r_c) * (t - 2) + ob_c_cost,
        # so inductively we can keep affording to buy obsidian robots, and at the end we're still left over with clay.
        o = min(o, max_o_cost + (max_o_cost - r_o) * (time - 1))
        c = min(c, ob_c_cost + (ob_c_cost - r_c) * (time - 1))
        ob = min(ob, g_ob_cost + (g_ob_cost - r_ob) * (time - 1))

        state = (r_o, r_c, r_ob, r_g, o, c, ob, g, time)

        if state in seen:
            continue

        seen.add(state)

        # For ore, clay, and obsidian robots, don't make more than we would possibly need. Note that
        # we can only make robot per turn, so the number of robots we need for ore/clay/obsidian should
        # be no more than the max amount of each resource that we would need to consume to make a robot.
        if o >= o_o_cost and r_o < max_o_cost:
            queue.append((
                r_o + 1, r_c, r_ob, r_g, o - o_o_cost + r_o, c + r_c, ob + r_ob, g + r_g, time - 1
            ))
        if o >= c_o_cost and r_c < ob_c_cost:
            queue.append((
                r_o, r_c + 1, r_ob, r_g, o - c_o_cost + r_o, c + r_c, ob + r_ob, g + r_g, time - 1
            ))
        if o >= ob_o_cost and c >= ob_c_cost and r_ob < g_ob_cost:
            queue.append((
                r_o, r_c, r_ob + 1, r_g, o - ob_o_cost + r_o, c - ob_c_cost + r_c, ob + r_ob, g + r_g, time - 1
            ))
        if o >= g_o_cost and ob >= g_ob_cost:
            queue.append((
                r_o, r_c, r_ob, r_g + 1, o - g_o_cost + r_o, c + r_c, ob - g_ob_cost + r_ob, g + r_g, time - 1
            ))

        queue.append((
            r_o, r_c, r_ob, r_g, o + r_o, c + r_c, ob + r_ob, g + r_g, time - 1
        ))

    return max_geodes

    

def main():
    with open("input.txt", 'r') as f:
        data = f.read().splitlines()
        blueprints = []
        for line in data:
            blueprint, ore, clay, obore, obclay, geore, geob = pat.parse(line)
            blueprints.append(Blueprint(blueprint, ore, clay, obore, obclay, geore, geob))
        ray.init()
        results = [get_max_geodes.remote(blueprint, 32) for blueprint in blueprints]
        results = ray.get(results)
        ray.shutdown()
        return reduce(operator.mul, results, 1)

if __name__ == '__main__':
    start_time = time.perf_counter()
    print(main())
    end_time = time.perf_counter()
    print(f'Time Elapsed: {end_time - start_time} seconds')

"""
Blueprint 1: Each ore robot costs 4 ore. Each clay robot costs 2 ore. Each obsidian robot costs 3 ore and 14 clay. Each geode robot costs 2 ore and 7 obsidian.
Blueprint 2: Each ore robot costs 2 ore. Each clay robot costs 3 ore. Each obsidian robot costs 3 ore and 8 clay. Each geode robot costs 3 ore and 12 obsidian.
"""