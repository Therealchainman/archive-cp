from typing import List
def radix_sort(leaderboard: List[int], equivalence_class: List[int]) -> List[int]:
    n = len(leaderboard)
    bucket_size = [0]*n
    for eq_class in equivalence_class:
        bucket_size[eq_class] += 1
    bucket_pos = [0]*n
    for i in range(1, n):
        bucket_pos[i] = bucket_pos[i-1] + bucket_size[i-1]
    updated_leaderboard = [0]*n
    for i in range(n):
        eq_class = equivalence_class[leaderboard[i]]
        pos = bucket_pos[eq_class]
        updated_leaderboard[pos] = leaderboard[i]
        bucket_pos[eq_class] += 1
    return updated_leaderboard

def suffix_array(s: str) -> List[int]:
    n = len(s)
    arr = [None]*n
    for i, ch in enumerate(s):
        arr[i] = (ch, i)
    arr.sort()
    leaderboard = [0]*n
    equivalence_class = [0]*n
    for i, (_, j) in enumerate(arr):
        leaderboard[i] = j
    equivalence_class[leaderboard[0]] = 0
    for i in range(1, n):
        left_segment = arr[i-1][0]
        right_segment = arr[i][0]
        equivalence_class[leaderboard[i]] = equivalence_class[leaderboard[i-1]] + (left_segment != right_segment)
    is_finished = False
    k = 1
    while k < n and not is_finished:
        for i in range(n):
            leaderboard[i] = (leaderboard[i] - k + n)%n # create left segment, keeps sort of the right segment
        leaderboard = radix_sort(leaderboard, equivalence_class) # radix sort for the left segment
        updated_equivalence_class = [0]*n
        updated_equivalence_class[leaderboard[0]] = 0
        for i in range(1, n):
            left_segment = (equivalence_class[leaderboard[i-1]], equivalence_class[(leaderboard[i-1]+k)%n])
            right_segment = (equivalence_class[leaderboard[i]], equivalence_class[(leaderboard[i]+k)%n])
            updated_equivalence_class[leaderboard[i]] = updated_equivalence_class[leaderboard[i-1]] + (left_segment != right_segment)
            is_finished &= (updated_equivalence_class[leaderboard[i]] != updated_equivalence_class[leaderboard[i-1]])
        k <<= 1
        equivalence_class = updated_equivalence_class
    return leaderboard

def main():
    s = input() + '$'
    return suffix_array(s)

if __name__ == '__main__':
    print(main())