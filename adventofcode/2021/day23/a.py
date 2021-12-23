
class Amphipod:
    def __init__(self):
        self.minCost = 1000000
        self.catalog = [10**i for i in range(4)]
    def data_loader(self):
        with open("inputs/input.txt", "r") as f:
            return f.read().splitlines()
    def dfs(self, locations, hallway_mask, siderooms, cost, count):
        if count == 0:
            print(self.minCost)
            self.minCost = min(self.minCost, cost)
            return
        # try to move the pods if can move
        for pod, locs in enumerate(locations):
            for j, (r, c) in enumerate(locs):
                # moving pods that are in incorrect side rooms into hallway or correct side room if available
                if r>1:
                    if r==2 or (r==3 and siderooms[c]==1):
                        current_cost = 2*self.catalog[pod] if r==3 else self.catalog[pod]
                        for i in range(c-1,0,-1):
                            if (hallway_mask>>i)&1 == 1:
                                break
                            if i != 3 and i != 5 and i != 7 and i != 9 and (hallway_mask>>i)&1 == 0:
                                hallway_mask |= 1<<i
                                siderooms[pod] -= 1
                                current_cost += self.catalog[pod]
                                locations[pod][j] = (1, i)
                                self.dfs(locations, hallway_mask, siderooms, cost+current_cost, count)
                                hallway_mask ^= (1<<i)
                                siderooms[pod] += 1
                                locations[pod][j] = (r,c)
                        current_cost = 2*self.catalog[pod] if r==3 else self.catalog[pod]
                        for i in range(r+1,12):
                            if (hallway_mask>>i)&1 == 1:
                                break
                            if i!=3 and i!=5 and i!=7 and i!=9 and (hallway_mask>>i)&1==0:
                                hallway_mask |= 1<<i
                                siderooms[pod] -= 1
                                current_cost += self.catalog[pod]
                                locations[pod][j] = (1, i)
                                self.dfs(locations, hallway_mask, siderooms, cost+current_cost, count)
                                hallway_mask ^= (1<<i)
                                siderooms[pod] += 1
                                locations[pod][j] = (r,c)
                else:
                    # moving pods that are in hallway to correct side room if available
                    for i, side in enumerate(range(3,10,2)):
                        print(i,side)
                        if pod==i and siderooms[side]==1:
                            siderooms[side]-=1
                            hallway_mask ^= 1<<c
                            print(f"before location: {locations}")
                            locations[pod].remove((r,c))
                            print(f"after location: {locations}")
                            self.dfs(locations, hallway_mask, siderooms, cost+self.catalog[pod], count-1)
                            locations[pod].append((r,c))
                            hallway_mask ^= 1<<c
                            siderooms[side]+=1


        
    def run(self):
        data = self.data_loader()
        amphipods = [[] for _ in range(4)]
        for r, line in enumerate(data):
            for c, char in enumerate(line):
                if char.isalpha():
                    amphipods[ord(char)-ord('A')].append((r, c))
        count_correct = 8
        siderooms = [2 for _ in range(10)]
        for pod, room in enumerate(range(3,10,2)):
            for depth in range(3,1,-1):
                if (depth, room) in amphipods[pod]:
                    amphipods[pod].remove((depth, room))
                    siderooms[room] -= 1
                    count_correct -= 1
                else:
                    break
        self.dfs(amphipods, 0, siderooms, 0, count_correct)

        

if __name__ == '__main__':
    a = Amphipod()
    a.run()


"""
Hallway [1,11]
"""