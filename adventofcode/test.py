#!/usr/bin/env python3

import pathlib
import sys

from typing import Iterable, Literal, Self

sys.path.append(str(pathlib.Path(__file__).resolve().parents[3] / 'lib' / 'python'))

import aoc
import aoc.search

DIR = (
  tuple[Literal[-1], Literal[0]] |
  tuple[Literal[1], Literal[0]] |
  tuple[Literal[0], Literal[-1]] |
  tuple[Literal[0], Literal[1]]
)

# row, column
EAST = (0, 1)
WEST = (0, -1)
NORTH = (-1, 0)
SOUTH = (1, 0)

Position = tuple[int, int]
RunPosition = tuple[int, int, DIR, int]

class CrucibleSearch(aoc.search.AStarSearch[RunPosition, Position]):
  __slots__ = ("grid", "bounds", "ultra")

  def __init__(self: Self, grid: str, ultra: bool = False, **kwargs) -> None:
    super().__init__(**kwargs)
    self.grid = grid.splitlines()
    self.bounds = (len(self.grid), len(self.grid[0]))
    self.ultra = ultra

  def is_finished(self: Self, pos: RunPosition, goal: Position) -> bool:
    r, c, _, run = pos
    return pos[:2] == goal and (not self.ultra or run >= 4)

  def estimate(self: Self, node: RunPosition, goal: Position) -> int:
    r, c, _, run = node
    remaining = aoc.manhattan((r, c), goal)
    if self.ultra:
      remaining += 4 - (min(run, 4))
    return remaining

  def moves(self: Self, node: RunPosition) -> Iterable[tuple[RunPosition, int]]:
    r, c, dir, run = node

    if self.ultra:
      dirs = [dir] if run < 10 else []
    else:
      dirs = [dir] if run < 3 else []

    if not self.ultra or run >= 4:
      dirs.extend([NORTH, SOUTH] if dir in {EAST, WEST} else [EAST, WEST])

    moves = []

    for ndir in dirs:
      nr, nc = (n + dn for n, dn in zip((r, c), ndir))
      if 0 <= nr < self.bounds[0] and 0 <= nc < self.bounds[1]:
        moves.append(((nr, nc, ndir, run + 1 if dir == ndir else 1), int(self.grid[nr][nc])))

    return moves

def run() -> None:
  with open(aoc.inputfile('big.txt')) as f:
    input = f.read().strip()

  goal = (input.count("\n"), input.find("\n") - 1)

  for ultra in (False, True):
    s = CrucibleSearch(input, ultra)
    _, heat, path = s.shortest_path((0, 0, EAST, 0), goal)
    print(f"Least heat loss{' (ultra)' if ultra else ''}: {heat}")

if __name__ == '__main__':
  run()
  sys.exit(0)
