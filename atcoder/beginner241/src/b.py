from collections import Counter

def can_eat():
  N, M = map(int, input().split())
  A = list(map(int,input().split()))
  B = list(map(int,input().split()))
  freq = Counter()
  for noodle_len in A:
    freq[noodle_len] += 1
  for desired_noodle in B:
    if freq[desired_noodle] == 0:
      return False
    freq[desired_noodle] -= 1
  return True

if __name__ == '__main__':
  if can_eat():
    print("Yes")
  else:
    print("No")
    

