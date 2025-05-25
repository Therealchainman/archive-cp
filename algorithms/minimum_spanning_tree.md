# Minimum Spanning Tree

Given an undirected weighted graph, a minimum spanning tree (MST) is a subset of the edges that connects all vertices together without any cycles and with the minimum possible total edge weight.

## All edges that are in at least one minimum spanning tree

Cut property: 
Let (S, V\S) be any cut of G and let e be a minimum‐weight edge in the cut‐edge set δ(S). Then there exists some Minimum Spanning Tree of G that contains e.

If you pick any partition of the vertices, the lightest edge crossing that partition is safe to add to your growing forest without ever “ruining” the possibility of ending with a global MST.

Cycle Property: For any cycle in the graph, the maximum‐weight edge on that cycle is not in any MST.


## Kruskal's Algorithm

sort edges in non-decreasing order of their weight
and add them into dsu if they don't form a cycle