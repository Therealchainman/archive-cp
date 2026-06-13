# Hopcroft Karp Algorithm

The Hopcroft-Karp algorithm is an efficient algorithm for finding a maximum matching in a bipartite graph.

A bipartite graph is a graph whose vertices can be split into two disjoint sets, usually called the left side and the right side, such that every edge connects a vertex on the left side to a vertex on the right side.

A matching is a set of edges where no two edges share an endpoint. In other words, each vertex can be matched with at most one other vertex. A maximum matching is a matching with the largest possible number of edges.

## Problem

Given a bipartite graph:

```text
Left side:  L_1, L_2, ..., L_n
Right side: R_1, R_2, ..., R_m
```

find the maximum number of pairs that can be selected such that:

```text
Each left vertex is used at most once.
Each right vertex is used at most once.
Only existing graph edges can be selected.
```

## Key idea

The algorithm starts with an empty matching and repeatedly improves it.

To improve a matching, it looks for an augmenting path.

An augmenting path is a path that:

```text
starts at an unmatched left vertex,
ends at an unmatched right vertex,
alternates between unmatched and matched edges.
```

If such a path is found, we can flip the status of every edge on the path:

```text
unmatched edges become matched,
matched edges become unmatched.
```

This increases the matching size by exactly one.

## Simple augmenting path algorithm

A simpler algorithm would find one augmenting path at a time and augment the matching by one.

That works, but it can be slow because it may need many separate searches.

If the maximum matching has size K, the simple algorithm may perform K augmentations.

## Hopcroft-Karp improvement

Hopcroft-Karp improves this by finding many shortest augmenting paths at once.

Each phase has two parts:

1. BFS builds layers of the graph from all currently unmatched left vertices.
2. DFS finds as many vertex-disjoint shortest augmenting paths as possible using those layers.

After all possible shortest augmenting paths are found, the algorithm augments along all of them in one phase.

This is faster than augmenting one path at a time.

## BFS phase

The BFS phase starts from every unmatched vertex on the left side.

It explores alternating paths:

```text
From a left vertex, it can use unmatched edges to right vertices.
From a right vertex, it follows the currently matched edge back to a left vertex.
```

The purpose of BFS is to compute distance layers, so that the later DFS only searches shortest augmenting paths.

If BFS cannot reach any unmatched right vertex, then no augmenting path exists and the current matching is maximum.

## DFS phase

After BFS builds the layers, DFS is run from each unmatched left vertex.

The DFS only follows edges that respect the BFS layering. This ensures that it only finds shortest augmenting paths.

Whenever DFS reaches an unmatched right vertex, it augments the matching along that path.

Because the DFS searches through the layered graph, one phase may find many augmenting paths.

## Why the algorithm is correct

A matching is maximum if and only if there is no augmenting path.

Hopcroft-Karp repeatedly finds augmenting paths and increases the matching size. Once BFS can no longer find any augmenting path, the algorithm stops. At that point, by the augmenting path theorem, the matching is maximum.

## Complexity

Let:

```text
V = number of vertices
E = number of edges
```

Hopcroft-Karp runs in:

```text
O(E sqrt(V))
```

This is faster than the basic augmenting-path algorithm, which is commonly:

```text
O(VE)
```

For large bipartite graphs, Hopcroft-Karp is usually the standard choice.

## Typical implementation data structures

A common implementation uses:

```text
adj[l]       = list of right vertices connected to left vertex l
matchL[l]   = right vertex matched to left vertex l, or -1 if unmatched
matchR[r]   = left vertex matched to right vertex r, or -1 if unmatched
dist[l]     = BFS layer distance for left vertex l
```

The algorithm then repeats:

```text
while BFS finds at least one augmenting path:
    for each unmatched left vertex:
        try DFS from that vertex
        if DFS succeeds:
            increase matching size
```

## High-level pseudocode

```text
initialize all left vertices and right vertices as unmatched
matching_size = 0

while BFS finds shortest augmenting path layers:
    for each left vertex l:
        if l is unmatched:
            if DFS(l) finds an augmenting path:
                matching_size += 1

return matching_size
```

## Intuition

The algorithm is efficient because it does not improve the matching one edge at a time blindly.

Instead, each round finds the current shortest way to improve the matching, then performs many independent improvements at once.

This batching of augmenting paths is the main reason Hopcroft-Karp is faster than the simpler DFS-based matching algorithm.
