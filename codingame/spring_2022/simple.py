import sys
import math
from heapq import heappush, heappop, heapify, nsmallest, nlargest
from collections import defaultdict
from itertools import product

# base_x: The corner of the map representing your base
base_x, base_y = [int(i) for i in input().split()]
heroes_per_player = int(input())  # Always 3
X=17630
Y=9000
enemy_x, enemy_y = X, Y
if not base_x==base_y==0:
  enemy_x, enemy_y = 0, 0

# game loop
while True:
    health_arr, mana_arr = [], []
    for i in range(2):
        health, mana = [int(j) for j in input().split()]
        health_arr.append(health)
        mana_arr.append(mana)
    entity_count = int(input()) 
    entities = defaultdict(dict)
    spiders = dict()
    my_player = []
    for i in range(entity_count):
        # _id: Unique identifier
        # _type: 0=monster, 1=your hero, 2=opponent hero
        # x: Position of this entity
        # shield_life: Ignore for this league; Count down until shield spell fades
        # is_controlled: Ignore for this league; Equals 1 when this entity is under a control spell
        # health: Remaining health of this monster
        # vx: Trajectory of this monster
        # near_base: 0=monster with no target yet, 1=monster targeting a base
        # threat_for: Given this monster's trajectory, is it a threat to 1=your base, 2=your opponent's base, 0=neither
        _id, _type, x, y, shield_life, is_controlled, health, vx, vy, near_base, threat_for = [int(j) for j in input().split()]
        entities[_type][_id] = {
          'x' : x,
          'y' : y,
          'shield_life' : shield_life,
          'is_controlled' : is_controlled,
          'health' : health,
          'vx' : vx,
          'vy' : vy,
          'near_base' : near_base,
          'threat_for' : threat_for
        }
        if _type == 0:
          spiders[_id] = entities[_type][_id]
        elif _type == 1:
          my_player.append(entities[_type][_id])
    # rank the spiders by threat for my lone defender
    spiders_rank = []
    if base_x == base_y == 0:
      x, y = X//4, Y//4
    else:
      x, y = int(X/1.4), int(Y/1.4)
    best = math.inf
    for spider in spiders.values():
      threat = 0
      if spider['near_base'] == 1 == spider['threat_for']:
        threat -= 3000
      threat += math.hypot(abs(base_x - spider['x']), abs(base_y - spider['y']))
      if threat < best:
        best = threat
        x, y = spider['x'], spider['y']
    for spider in spiders.values():
      threat = 0
      threat += math.hypot(abs(enemy_x - spider['x']), abs(enemy_y - spider['y']))
      if 5000 <= math.hypot(abs(enemy_x-spider['x']), abs(enemy_y-spider['y'])) <= 7000:
        threat -= 3000
      spiders_rank.append((threat, (spider['x'], spider['y'])))
    heapify(spiders_rank)
    attk_wind = mana_arr[0] >= 150
    def_wind = False
    # SET THE DISTANCE AT WHICH TO USE WIND TO DEFEND BASE
    if math.hypot(abs(base_x-x), abs(base_y-y)) <= 3000 and math.hypot(abs(x-my_player[0]['x']), abs(y-my_player[0]['y'])) <= 1280:
      # use wind to defend base
      def_wind = True
    for i in range(heroes_per_player):
        # In the first league: MOVE <x> <y> | WAIT; In later leagues: | SPELL <spellParams>;
        if i == 0: # defender
          if def_wind:
            print(f"SPELL WIND {X//2} {Y//2}")
          else:
            print(f"MOVE {x} {y}")
        else:
          x, y = enemy_x, enemy_y
          player_x, player_y = my_player[i]['x'], my_player[i]['y']
          dist_enemy = math.hypot(abs(player_x-enemy_x), abs(player_y-enemy_y))
          if dist_enemy <= 5000 or dist_enemy>10000:
            if enemy_x==enemy_y==0:
              x, y = X//3, Y//3
            else:
              x, y = int(X/1.5), int(Y/1.5)
          elif spiders_rank:
            _, (x, y) = heappop(spiders_rank)
          if attk_wind:
            print(f"SPELL WIND {enemy_x} {enemy_y}")
          else:
            print(f"MOVE {x} {y}")