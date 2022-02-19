"""Starter Code for Milk Tea CC Problem"""

# Complete the count_complaints function
def count_complaints(preferences, forbiddens):
  complaints = 0
  # TODO: Add logic to count the number of complaints
  return complaints

if __name__ == '__main__':
  # Read number of test cases
  num_cases = int(input())

  for tc in range(1, num_cases + 1):
    # Read number of friends, number of forbidden teas, and number of options
    num_friends, num_forbidden, num_options = map(int, input().split())

    # Read the friends' preferences
    preferences = [input() for _ in range(num_friends)]

    # Read the forbidden teas
    forbiddens = [input() for _ in range(num_forbidden)]

    print("Case #%d: %d" % (tc, count_complaints(preferences, forbiddens)))
