

class Solve:
  def __init__(self):
    self.N = int(input())
    self.grid = [input() for _ in range(self.N)]
  
  def vertical(self):
    for c in range(self.N):
      cnt_white = 0
      for r in range(self.N):
        cnt_white += (self.grid[r][c]=='.')
        if r > 4:
          if cnt_white <= 2: return True
          cnt_white -= (self.grid[r-5][c] == '.')
    return False

  def horizontal(self):
    for r in range(self.N):
      cnt_white = 0
      for c in range(self.N):
        cnt_white += (self.grid[r][c]=='.')
        if c > 4:
          if cnt_white <= 2: return True
          cnt_white -= (self.grid[r][c-5] == '.')
    return False

  def in_bounds(self, row, col):
    return 0<=row<self.N and 0<=col<self.N

  def diagonal(self):
    # MAIN DIAGONALS (LEFT TO RIGHT)
    for c in range(self.N):
      cnt_white = 0
      for i in range(self.N):
        if not self.in_bounds(i,c+i): break
        cnt_white += (self.grid[i][c+i]=='.')
        if i > 4:
          if cnt_white <= 2: return True
          cnt_white -= (self.grid[i-5][c+i-5]=='.')
    for r in range(1, self.N):
      cnt_white = 0
      for i in range(self.N):
        if not self.in_bounds(r+i,i): break
        cnt_white += (self.grid[r+i][i]=='.')
        if i > 4:
          if cnt_white <= 2: return True
          cnt_white -= (self.grid[r+i-5][i-5]=='.')

    # MINOR DIAGONALS (RIGHT TO LEFT)
    for c in range(self.N):
      cnt_white = 0
      for i in range(self.N):
        if not self.in_bounds(i,c-i): break
        cnt_white += (self.grid[i][c-i]=='.')
        if i > 4:
          if cnt_white <= 2: return True
          cnt_white -= (self.grid[i-5][c-i+5]=='.')
    for r in range(1, self.N):
      cnt_white = 0
      c = self.N - 1
      for i in range(self.N):
        if not self.in_bounds(r+i,c-i): break
        cnt_white += (self.grid[r+i][c-i]=='.')
        if i > 4:
          if cnt_white <= 2: return True
          cnt_white -= (self.grid[r+i-5][c-i+5]=='.')
    return False

  def main(self):
    if self.vertical() or self.horizontal() or self.diagonal(): return True
    return False

if __name__ == '__main__':
  sol = Solve()
  if sol.main():
    print("Yes")
  else:
    print("No")