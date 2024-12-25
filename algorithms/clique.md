# Clique

## Bron-Kerbosch algorithm to enumerate maximal cliques in undirected graph

This algorithm is really 3^(n/3) time complexity but that is worse case for when it is operating on a dense graph.  For a sparse graph it is very fast in practice, as long as it includes pivoting, as this example does. 

If you need to find the maximum clique, you can use this as well, just get the largest maximal clique.

Maximal clique just means it can not be enlarged with the current set of vertices. 

```py
def bron_kerbosch(R, P, X):
    if not P and not X: yield R
    pivot = next(iter(P | X)) if P or X else None
    non_neighbors = P - adj[pivot] if pivot else P

    for v in non_neighbors:
        yield from bron_kerbosch(R | {v}, P & adj[v], X & adj[v])
        P.remove(v)
        X.add(v)
```

Without pivot and returning all cliques of size 3, can adjust to any size.  You have to remove pivot optimization to do this. 

```py
def bron_kerbosch(R, P, X):
    if len(R) == 3: 
        yield R
        return
    if not P and not X: yield R
    for v in P.copy():
        yield from bron_kerbosch(R | {v}, P & adj[v], X & adj[v])
        P.remove(v)
        X.add(v)
```