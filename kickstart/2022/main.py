from heapq import *
        
def main():
    D, N, X = map(int, input().split())
    seeds = [None]*N
    for i in range(N):
        Q, L, V = map(int, input().split())
        seeds[i] = (L, Q, V)
    seeds.sort()
    max_heap = [] # (value, seed index)
    cur_day = res = seed_index = remaining_harvest_cur_day = 0 # counting backwards from day D
    while cur_day < D:
        if not max_heap and seed_index == N: # planted all and harvested all seeds
            break
        if seed_index < N and seeds[seed_index][0] == cur_day: # push some more seeds into the max heap
            maturation_time, quantity, value = seeds[seed_index]
            heappush(max_heap, (-value, -quantity))
            seed_index += 1
        elif not max_heap: # if no seeds to plant, move to the next day when you can plant a seed that you will be able to harvest on time
            maturation_time, quantity, value = seeds[seed_index]
            heappush(max_heap, (-value, -quantity))
            seed_index += 1
            remaining_harvest_cur_day = 0
            cur_day = maturation_time
        elif remaining_harvest_cur_day == 0: # harvest the seeds with the full day available
            value, quantity = map(abs, heappop(max_heap))
            max_days = min(D, seeds[seed_index][0]) if seed_index < N else D # it has at least maximum value up to max_days
            days_to_harvest = max_days - cur_day
            harvest = min(X * days_to_harvest, quantity)
            res += harvest * value
            num_days_to_harvest = harvest//X
            cur_day += num_days_to_harvest
            remaining_harvest_cur_day = 0
            if harvest % X != 0:
                remaining_harvest_cur_day = X - harvest % X
            if quantity - harvest > 0: # if there are still seeds left, push them back into the max heap
                heappush(max_heap, (-value, -(quantity - harvest)))
        else: # remaining_harvest_cur_day > 0
            value, quantity = map(abs, heappop(max_heap))
            harvest = min(remaining_harvest_cur_day, quantity)
            res += harvest * value
            remaining_harvest_cur_day -= harvest
            if quantity - harvest > 0: # if there are still seeds left, push them back into the max heap
                heappush(max_heap, (-value, -(quantity - harvest)))
            if remaining_harvest_cur_day == 0:
                cur_day += 1
    return res

if __name__ == '__main__':
    T = int(input())
    for t in range(1, T+1):
        print(f'Case #{t}: {main()}')