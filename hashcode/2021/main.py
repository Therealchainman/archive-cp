import sys

# reading inputs

class Car:
  pass

# NODE IN THE GRAPH
class Intersection:
  def __init__(self, intersection_id, edges = [], neighbors = []):
    self.intersection_id = intersection_id
    self.neighbors = neighbors
    self.edges = edges

# EDGE IN THE GRAPH
class Street:
  def __init__(self, start, end, name, time):
    self.source_intersection = start
    self.target_intersection = end
    self.street_id = name
    self.length = time

# CONTAINS THE GRAPH OF INTERSECTIONS AND STREETS
class Path:
  def __init__(self):
    self.intersections = {}

class Solver:  
  def data_loader(self):
    with open('inputs/a.txt') as f:
      self.duration, self.num_intersections, self.num_streets, self.num_cars, self.bonus = map(int,f.readline().split())
      self.path = Path()
      self.streets = {}
      self.intersections = {} # map intersection id to neighbors
      for _ in range(self.num_streets):
        start, end, name, time = map(int, f.readline().split())
        if name not in self.streets:
          self.streets[name] = Street(start, end, name)
        self.streets.append(Street(start, end, name, time))


if __name__ == '__main__':
  sol = Solver()
  sol.data_loader()