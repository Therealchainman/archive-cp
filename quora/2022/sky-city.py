"""
basic algorithm is tree traversal with a minheap
"""
import heapq
from collections import namedtuple
import math

class SkyCity:
    def data_loader(self):
        # with open("inputs/input.txt", "r") as f:
        #     self.n = int(f.readline())
        #     self.graph = [[] for _ in range(self.n+1)]
        #     for _ in range(self.n-1):
        #         u, v, w = map(int, f.readline().split())
        #         self.graph[u].append((v, w))
        #         self.graph[v].append((u, w))
            
        #     Charge = namedtuple("Charge", ["cost", "rent"])
        #     self.data = [Charge(0,0)]
        #     for i in range(1,self.n+1):
        #         self.data.append(Charge(*map(int, f.readline().split())))
        self.n = int(input())
        self.graph = [[] for _ in range(self.n+1)]
        for _ in range(self.n-1):
            u, v, w = map(int, input().split())
            self.graph[u].append((v, w))
            self.graph[v].append((u, w))
        Charge = namedtuple("Charge", ["cost", "rent"])
        self.data = [Charge(0,0)]
        for i in range(1,self.n+1):
            self.data.append(Charge(*map(int, input().split())))

    def run(self):
            self.data_loader()
            heap = []
            dist = [math.inf for _ in range(self.n+1)]
            heapq.heappush(heap,(self.data[1].rent,1,self.data[1].cost, -1))
            dist[1]=1
            while heap:
                d, u, cost, parent = heapq.heappop(heap)
                for v, w in self.graph[u]:
                    if v==parent: continue
                    con_cost = d + cost*w
                    start_cost = d + self.data[u].cost*w+self.data[u].rent
                    if con_cost < dist[v]:
                        dist[v] = con_cost
                    heapq.heappush(heap,(con_cost,v,cost,u))
                    if start_cost < dist[v]:
                        dist[v] = start_cost
                    heapq.heappush(heap,(start_cost,v,self.data[u].cost,u))
            print(" ".join(map(str,dist[2:])))
if __name__ == '__main__':
    SkyCity().run()
