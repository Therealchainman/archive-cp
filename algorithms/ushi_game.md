# Ushi Game

In the Ushi game, each move imposes a constraint on state variables (e.g. resource levels or timing differences) that can be written in the form:

x_j - x_i ≤ c_ij

These **difference constraints** define a directed, weighted graph:

- **Vertices** represent the variables (e.g. game states or timestamps)
- **Edges** of the form `(i → j)` carry weights `c_ij` representing the constraint

Finding an optimal strategy reduces to computing shortest paths (or detecting infeasibility via negative cycles) in this graph — which is exactly what the **Bellman–Ford algorithm** solves in `O(VE)` time.

Equivalently, one can view the problem as a special **linear programming (LP)** instance: maximize or minimize a linear objective, subject to difference constraints. This can be solved using the same graph-based methods.

### Key Points

- **Graph construction**: how moves map to edges and weights  
- **Bellman–Ford relaxation**: iteratively tightening bounds  
- **Negative-cycle detection**: identifying infeasible or “infinite-gain” situations  
- **LP equivalence**: understanding why solving the LP reduces to shortest paths