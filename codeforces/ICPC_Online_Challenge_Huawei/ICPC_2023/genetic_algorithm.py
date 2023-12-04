import os,sys
from io import BytesIO, IOBase
from typing import *
 
# Fast IO Region
BUFSIZE = 8192
class FastIO(IOBase):
    newlines = 0
    def __init__(self, file):
        self._fd = file.fileno()
        self.buffer = BytesIO()
        self.writable = "x" in file.mode or "r" not in file.mode
        self.write = self.buffer.write if self.writable else None
    def read(self):
        while True:
            b = os.read(self._fd, max(os.fstat(self._fd).st_size, BUFSIZE))
            if not b:
                break
            ptr = self.buffer.tell()
            self.buffer.seek(0, 2), self.buffer.write(b), self.buffer.seek(ptr)
        self.newlines = 0
        return self.buffer.read()
    def readline(self):
        while self.newlines == 0:
            b = os.read(self._fd, max(os.fstat(self._fd).st_size, BUFSIZE))
            self.newlines = b.count(b"\n") + (not b)
            ptr = self.buffer.tell()
            self.buffer.seek(0, 2), self.buffer.write(b), self.buffer.seek(ptr)
        self.newlines -= 1
        return self.buffer.readline()
    def flush(self):
        if self.writable:
            os.write(self._fd, self.buffer.getvalue())
            self.buffer.truncate(0), self.buffer.seek(0)
class IOWrapper(IOBase):
    def __init__(self, file):
        self.buffer = FastIO(file)
        self.flush = self.buffer.flush
        self.writable = self.buffer.writable
        self.write = lambda s: self.buffer.write(s.encode("ascii"))
        self.read = lambda: self.buffer.read().decode("ascii")
        self.readline = lambda: self.buffer.readline().decode("ascii")
sys.stdin, sys.stdout = IOWrapper(sys.stdin), IOWrapper(sys.stdout)
input = lambda: sys.stdin.readline().rstrip("\r\n")

# IMPORTS
from itertools import product
import random
import math
import time

# sys.stdin = open("tests/06", "r")
# sys.stdout = open("output.txt", "w")

# READ IN DATA
N = int(input())
K = int(input()) # 10
T = int(input()) # 200
R = int(input()) # 5

# CONSTANTS
init_SINR = [[[[0] * N for _ in range(R)] for _ in range(K)] for _ in range(T)]
interference = [[[[0] * N for _ in range(N)] for _ in range(R)] for _ in range(K)]

for t, k, r in product(range(T), range(K), range(R)):
    arr = list(map(float, input().split()))
    for n in range(N):
        init_SINR[t][k][r][n] = arr[n]
for k, r, m in product(range(K), range(R), range(N)):
    arr = list(map(float, input().split()))
    for n in range(N):
        interference[k][r][n][m] = arr[n]
print(init_SINR)
active_users = [[] for _ in range(T)]
J = int(input())
frames = [None] * J
for j in range(J):
   arr = list(map(int, input().split()))
   TBS, user_id, start, length = arr[1:]
   end = start + length - 1
   frames[j] = (user_id, TBS, start, end)
frames.sort()
for (n, _, start, end) in frames:
    for t in range(start, end + 1):
        active_users[t].append(n)

# EQUATIONS
# time, spatial, frequency, user
# T, K, R, N = 1, 2, 3, 4
W = 192
def snr(t, k, r, n, power):
    # directly proportional to S and P for current and the the interference with all the other users in this block.
    blocks_other_interference = math.prod(math.exp(interference[k][r][n][m] * active(t, k, r, n, power)) for m in range(N) if m != n)
    initial_channel_signal = init_SINR[t][k][r][n]
    power_given = power[t][k][r][n]
    numerator = initial_channel_signal * power_given * blocks_other_interference
    # print(f"S: {initial_channel_signal}")
    # print(f"P: {power_given}")
    # print(f"interference from others in this block: {blocks_other_interference}")
    denominator = 1 + sum(init_SINR[t][kp][r][n] * power[t][kp][r][m] * math.exp(-interference[kp][r][n][m]) for kp, m in product(range(K), range(N)) if kp != k and m != n)
    return numerator / denominator
def active(t, k, r, n, power):
    return 1.0 if power[t][k][r][n] > 0 else 0.0
# keep note that a geometric mean means it will be a little bit resilient to outliers, but still they affect a geometric mean
# it is just what would be the factors if they were all the same and the product equal to the total product.
def snr_geometric_mean(t, k, n, power):
    try:
        return pow(math.prod(snr(t, k, r, n, power) for r in range(R) if active(t, k, r, n, power)), 1 / sum([active(t, k, r, n, power) for r in range(R)]))
    except:
        return 0
def shannon_capacity(t, k, n, power):
    return math.log2(1 + snr_geometric_mean(t, k, n, power))
def transmission_bits(start, end, n, power):
    # st = time.perf_counter()
    res = W * sum(active(t, k, r, n, power) * shannon_capacity(t, k, n, power) for t, k, r in product(range(start, end + 1), range(K), range(R)))
    # et = time.perf_counter()
    # print("transmission_bits took: ", et - st)
    return res
def score(power):
    transmitted = sum(1 for user_id, TBS, start, end in frames if transmission_bits(start, end, user_id, power) >= TBS)
    total = 0
    for t, k in product(range(T), range(K)):
        psum = 0
        for r, n in product(range(R), range(N)):
            psum += power[t][k][r][n]
            total += power[t][k][r][n]
            assert power[t][k][r][n] <= 4, print(f"P[t][k][r][n] exceeds 4: t = {t}, k = {k}, r = {r}, n = {n}, P[t][k][r][n] = {power[t][k][r][n]}")
        assert psum <= R, print(f"sum(P[t][k]) exceeded R: t = {t}, k = {k}, psum = {psum}")
    # print("transmitted", transmitted, "power_sum", power_sum)
    return transmitted - pow(10, -6) * total

# GENETIC ALGORITHM
class GeneticAlgorithm:
    def __init__(self, pop_size, num_generations, mutation_rate, tournament_size):
        self.pop_size = pop_size
        self.num_generations = num_generations
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        self.population = [None] * pop_size
        self.cell_power = [{} for _ in range(pop_size)]

    def split_integer_into_floats(self, R, N):
        # Generate N-1 random floats between 0 and R
        floats = [random.uniform(0, R) for _ in range(R * N + 1)]
        
        # Sort the floats
        floats.sort()
        
        # Calculate the differences between consecutive floats
        differences = [min(floats[i + 1] - floats[i], 3.99) for i in range(R * N)]
        return differences

    # create Individual (dna strand)
    def create_individual(self, idx):
        # st = time.perf_counter()
        power = [[[[0 for _ in range(len(active_users[t]))] for _ in range(R)] for _ in range(K)] for t in range(T)]
        # et = time.perf_counter()
        # st = time.perf_counter()
        # print("creating power array took: ", et - st)
        for t, k in product(range(T), range(K)):
            partitions = self.split_integer_into_floats(R, len(active_users[t]))
            partition_sum = 0
            for i, (r, n) in enumerate(product(range(R), range(len(active_users[t])))):
                power[t][k][r][n] = partitions[i]
                partition_sum += partitions[i]
            self.cell_power[idx][(t, k)] = partition_sum
            assert partition_sum <= R, f"sum of partitions: {partition_sum}"
            # print("idx", idx, "t", t, "k", k, "sum", partition_sum)
        # et = time.perf_counter()
        # print("creating partitions took: ", et - st)
        return power

    def create_population(self):
        for i in range(self.pop_size):
            self.population[i] = self.create_individual(i)

    # Calculate the fitness of a solution (the score)
    # this is the one that will evaluate the complicated function
    # TODO: write the equation to calcualte the score
    def calculate_fitness(self, index):
        # compute calculated function
        return score(self.population[index])

    # tournament selection
    def selection(self, tournament_size):
        idx1 = random.randint(0, self.pop_size - 1)
        idx2 = random.randint(0, self.pop_size - 1)
        return max([idx1, idx2], key = self.calculate_fitness)

    def crossover(self, p1, p2, idx1, idx2):
        offspring1 = [[[[0 for _ in range(len(active_users[t]))] for _ in range(R)] for _ in range(K)] for t in range(T)]
        offspring2 = [[[[0 for _ in range(len(active_users[t]))] for _ in range(R)] for _ in range(K)] for t in range(T)]
        mid = random.randint(1, T * K - 1)
        for i, (t, k) in enumerate(product(range(T), range(K))):
            self.cell_power[idx1][(t, k)] = 0
            self.cell_power[idx2][(t, k)] = 0
            for r, n in product(range(R), range(len(active_users[t]))):
                if i < mid:
                    offspring1[t][k][r][n] = self.population[p1][t][k][r][n]
                    offspring2[t][k][r][n] = self.population[p2][t][k][r][n]
                    self.cell_power[idx1][(t, k)] += self.population[p1][t][k][r][n]
                    self.cell_power[idx2][(t, k)] += self.population[p2][t][k][r][n]
                else:
                    offspring1[t][k][r][n] = self.population[p2][t][k][r][n]
                    offspring2[t][k][r][n] = self.population[p1][t][k][r][n]
                    self.cell_power[idx1][(t, k)] += self.population[p2][t][k][r][n]
                    self.cell_power[idx2][(t, k)] += self.population[p1][t][k][r][n]
        return offspring1, offspring2

    def mutate(self, ind):
        for t, k, r in product(range(T), range(K), range(R)):
            for n in range(len(active_users[t])):
                if random.random() < self.mutation_rate:
                    self.cell_power[ind][(t, k)] -= self.population[ind][t][k][r][n]
                    upper_bound = max(0, min(R - self.cell_power[ind][(t, k)] - 0.05, 3.99))
                    new_power = random.uniform(0, upper_bound)
                    self.population[ind][t][k][r][n] = new_power
                    self.cell_power[ind][(t, k)] += new_power

    def satisfies(self, ind, pp, stage):
        for t, k in product(range(T), range(K)):
            s = 0
            for r, n in product(range(R), range(N)):
                s += pp[t][k][r][n]
            # if s != self.cell_power[ind][(t, k)]: 
            #     print(f"mismatched: {stage}")
            #     print("ind", ind, "s", s, "t", t, "k", k, self.cell_power[ind][(t, k)])
            # else:
            #     print("good at stage: ", stage)
            if s > R: 
                print(f"unsatisfactory at stage: {stage}")
                print("s", s, self.cell_power[ind][(t, k)])

    # TODO: keep the best solution from all generations, in case it flips that best solution.
    def run(self):
        # population
        self.create_population()
        sc = -math.inf
        for i in range(self.pop_size):
            sc = max(sc, self.calculate_fitness(i))
        first = prev = sc
        # for i in range(self.pop_size):
        #     self.satisfies(i, self.population[i], "initial")
        for gi in range(self.num_generations):
            parents = []
            # selection
            for _ in range(self.pop_size):
                parents.append(self.selection(self.tournament_size))
            offsprings = [None] * self.pop_size
            # crossover
            for i in range(1, len(parents), 2):
                offsprings[i - 1], offsprings[i] = self.crossover(parents[i - 1], parents[i], i - 1, i)
                # self.satisfies(i - 1, offspring1, "crossover")
                # self.satisfies(i, offspring2, "crossover")
            self.population = offsprings
            # mutation
            for i in range(self.pop_size):
                self.mutate(i)
                # self.satisfies(i, self.population[i], "mutate")
            sc = -math.inf
            for i in range(self.pop_size):
                sc = max(sc, self.calculate_fitness(i))
            print(f"best solution after generation: {gi + 1}, fitness score: {sc}, delta: {sc - prev}")
            prev = sc
            # Output the final best solution
        # sc = -math.inf
        # best_idx = 0
        # for i in range(self.pop_size):
        #     cur = self.calculate_fitness(i)
        #     if cur > sc:
        #         sc = cur
        #         best_idx = i
        # print("\nFinal Best Solution:", sc)
        # # print(f"gain: {sc - first}")
        # return self.population[best_idx]
        return self.population[0]

def main():
    # PARAMETERS
    pop_size = 4
    num_generations = 10
    mutation_rate = 0.05
    # mutation_rate = 0.005
    tournament_size = 2
    GA = GeneticAlgorithm(pop_size, num_generations, mutation_rate, tournament_size)
    ans = GA.run()
    for t, k, r in product(range(T), range(K), range(R)):
        ptr = 0
        for n in range(N):
            if ptr < len(active_users[t]) and active_users[t][ptr] == n:
                print(ans[t][k][r][ptr], end = " ")
                ptr += 1
            else:
                print(0, end = " ")
        print()

if __name__ == "__main__":
    main()