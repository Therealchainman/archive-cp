# TODO: Complete the get_ruler function
def get_ruler(kingdom):
  ruler = ''
  # TODO: Add logic to determine the ruler of the kingdom
  # It should be either 'Alice', 'Bob' or 'nobody'.
  vowels = ['a', 'e', 'i', 'o', 'u']
  if kingdom[-1].lower() in vowels:
    ruler = 'Alice'
  elif kingdom[-1].lower() == 'y':
    ruler = 'nobody'
  else:
    ruler = 'Bob'
  return ruler

def main():
  # Get the number of test cases
  T = int(input())
  for t in range(T):
    # Get the kingdom
    kingdom = input()
    print('Case #%d: %s is ruled by %s.' % (t + 1, kingdom, get_ruler(kingdom)))

if __name__ == '__main__':
  main()
