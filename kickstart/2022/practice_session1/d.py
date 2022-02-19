IMP = "Impossible"
BLU = "Blue wins"
RED = "Red wins"
NON = "Nobody wins"
# TODO: Optimize algorithm to pass test set 2
# TODO: Implement tarjan's algorithm (DFS tree) for finding articulation points
from collections import deque, defaultdict
from copy import deepcopy
import sys

# sys.stdout = open('outputs/output.txt', 'w')
class Hex:
  def __init__(self, board_size, board):
    self.N = board_size
    self.board = board
    self.neighborhood = ((1,0),(-1,0),(0,-1),(0,1),(1,-1),(-1,1))
  
  def init_for_articulation(self, N):
    self.disc = [0 for _ in range(N)]
    self.low = [0 for _ in range(N)]
    self.time = 0
    self.articulation_points = [0 for _ in range(N)]

  def in_bounds(self, r, c):
    return 0<=r<self.N and 0<=c<self.N

  def build_graphB(self, color):
    self.graphB = defaultdict(list)
    get_vertex = {}
    n = 0
    queue = deque([])
    for r in range(self.N):
      if self.board[r][0] == color:
        n += 1
        self.graphB[0].append(n)
        self.graphB[n].append(0)
        get_vertex[(r,0)] = n
        queue.append((n, r, 0))
        self.board[r][0] = color.lower()
    while len(queue)>0:
      vertex, r, c = queue.popleft()
      for dr, dc in self.neighborhood:
        nr, nc = r + dr, c + dc
        if not self.in_bounds(nr,nc): continue
        if self.board[nr][nc] == color:
          n += 1
          get_vertex[(nr,nc)] = n
          queue.append((n, nr, nc))
          self.board[nr][nc] = color.lower()
        u = get_vertex[(nr,nc)]
        self.graphB[u].append(vertex)
        self.graphB[vertex].append(u)

  def build_graphR(self, color):
    self.graphR = defaultdict(list)
    get_vertex = {}
    n = 0
    queue = deque([])
    for c in range(self.N):
      if self.board[0][c] == color:
        n += 1
        self.graphR[0].append(n)
        self.graphR[n].append(0)
        get_vertex[(0,c)] = n
        queue.append((n, 0, c))
        self.board[0][c] = color.lower()
    while len(queue)>0:
      vertex, r, c = queue.popleft()
      for dr, dc in self.neighborhood:
        nr, nc = r + dr, c + dc
        if not self.in_bounds(nr,nc): continue
        if self.board[nr][nc] == color:
          n += 1
          get_vertex[(nr,nc)] = n
          queue.append((n, nr, nc))
          self.board[nr][nc] = color.lower()
        u = get_vertex[(nr,nc)]
        self.graphR[u].append(vertex)
        self.graphR[vertex].append(u)

  def get_articulation_points(self, vertex, parent):
    children = 0
    self.time += 1
    self.low[vertex] = self.disc[vertex] = self.time
    for nei in self.graph[vertex]: # nei has not been discovered before
      if nei==parent: continue
      if self.disc[nei]>0:
        children += 1
        self.get_articulation_points(nei, vertex)
        if self.disc[vertex] <= self.low[nei]:
          self.articulation_points[vertex] = 1
        self.low[vertex] = min(self.low[nei], self.low[vertex])
      else: # we have found an ancestor it already discovered
        self.low[vertex] = min(self.low[vertex], self.disc[nei])
      
  def valid_articulation_point(self, vertex, color):
    pass

  def game_status(self, board_size, board):
    count_blue, count_red = sum(row.count('B') for row in board), sum(row.count('R') for row in board)
    if abs(count_blue-count_red)>1: return IMP
    
    # CHECK PLAYER BLUE
    self.build_graphB('B')
    self.init_for_articulation(len(self.graphB))
    self.get_articulation_points(0, None)
    blue_wins = False
    for vertex in self.graphB.keys():
      if self.articulation_points[vertex]==1 and self.valid_articulation_point(vertex, 'B'):
        blue_wins = True
        break
    if blue_wins and count_red>count_blue: return IMP
    if blue_wins: return BLU

    # CHECK PLAYER RED
    self.build_graphR('R')
    self.init_for_articulation(len(self.graphR))
    self.get_articulation_points(0, None)
    red_wins = False
    for vertex in self.graphR.keys():
      if self.articulation_points[vertex]==1 and self.valid_articulation_point(vertex, 'R'):
        red_wins = True
        break
    if red_wins and count_blue>count_red: return IMP
    if red_wins: return RED

    return NON

def main():
  with open("inputs/input1.txt", "r") as f:
    test_cases = int(f.readline())
    for test_case in range(1, test_cases + 1, 1):
      board_size = int(f.readline())
      board = []
      for _ in range(board_size):
        board.append(list(f.readline().strip()))
      hex_game = Hex(board_size, board)
      ans = hex_game.game_status(board_size, board)

      print("Case #{}: {}".format(test_case, ans))

  # test_cases = int(input())
  # for test_case in range(1, test_cases + 1, 1):
  #   board_size = int(input())
  #   board = []
  #   for _ in range(board_size):
  #     board.append(list(input().strip()))

  #   ans = game_status(board_size, board)

  #   print("Case #{}: {}".format(test_case, ans))

if __name__ == "__main__":
  main()
  # sys.stdout.close()
