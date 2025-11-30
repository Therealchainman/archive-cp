# EXPECTED VALUE

Informally, the expected value is the arithmetic mean of a large number of independently selected outcomes of a random variable.

## EXPECTED MAXIMUM EXAMPLE PROBLEM

Twilight Sparkle was playing Ludo with her friends Rainbow Dash, Apple Jack and Flutter Shy. But she kept losing. Having returned to the castle, Twilight Sparkle became interested in the dice that were used in the game.

The dice has m faces: the first face of the dice contains a dot, the second one contains two dots, and so on, the m-th face contains m dots. Twilight Sparkle is sure that when the dice is tossed, each face appears with probability . Also she knows that each toss is independent from others. Help her to calculate the expected maximum number of dots she could get after tossing the dice n times.

$E(x) = $

```py
m, n = map(int, input().split())
prob = lambda x: pow((x / m), n)
res = sum(i * (prob(i) - prob(i - 1)) for i in range(1, m + 1))
print(res)  
```

## Expected value for indicator random variables

Indicator random variables describe experiments to detect whether or not something happened.

The expected value of an indicator random variable for an event is just the probability of that
event. (Remember that a random variable IA is the indicator random variable for event A, if
IA = 1 when A occurs and IA = 0 otherwise.)

$E[IA] = Pr{A}$

moving robots example problem

```py
"""
probability cell is empty is 1 - probability of robot existing on cell
"""
from itertools import product

def main():
    k = int(input())
    n = 8
    board = [[[0] * n for _ in range(n)] for _ in range(n * n)]
    neighborhood = lambda r, c: [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
    in_bounds = lambda r, c: 0 <= r < n and 0 <= c < n
    for i in range(n * n):
        r, c = i // n, i % n
        board[i][r][c] = 1
    on_corner = lambda r, c: (r == 0 and c == 0) or (r == 0 and c == n - 1) or (r == n - 1 and c == 0) or (r == n - 1 and c == n - 1)
    on_boundary = lambda r, c: r == 0 or r == n - 1 or c == 0 or c == n - 1
    for _ in range(k):
        nboard = [[[0] * n for _ in range(n)] for _ in range(n * n)]
        for i, r, c in product(range(n * n), range(n), range(n)):
            p = 3 if on_boundary(r, c) else 4
            p = 2 if on_corner(r, c) else p
            for nr, nc in neighborhood(r, c):
                if in_bounds(nr, nc):
                    nboard[i][nr][nc] += board[i][r][c] / p
        board = nboard
    """
    probability that first robot is not in that cell at kth step, 1 - probability robot exists in that cell at kth step
    so it should be multiplied, because you want the probability of the sequence that robot1, robot2, robot3 are all not at that cell
    so how to do this for all.
    low*high = low 
    low*low = low
    high*high = high
    """
    res = [[1] * n for _ in range(n)]
    for i, r, c in product(range(n * n), range(n), range(n)):
        res[r][c] *= (1 - board[i][r][c])
    """
    expectation value is sum of all probabilities of each cell
    using linearity of expectation
    E[x+y] = E[x] + E[y
    that is expecation value of all cells is equal to expectation value of each cell that it is empty
    """
    sum_ = sum(sum(row) for row in res)
    print(f"{sum_:0.6f}")

if __name__ == '__main__':
    main()
```

## Expected Value with Recurrence

The expected value can also be computed using a recurrence relation. This is particularly useful in dynamic programming problems where the expected value of a state can be expressed in terms of the expected values of previous states.

For these you usually have to define some states and the state transitions. 

They are using related to the number of steps to complete a process

Here’s a general template for solving expected-value problems with state transitions, especially of the kind where you simulate a process until some stopping condition (like your sock problem). 

### Step 1: Define the States Clearly
What does a “state” represent?
- A specific configuration of the process (e.g., which sock you're holding)
- States should be small and uniquely representable (e.g., by an index)

### Step 2: Determine the Transitions
From each state, list out:
- What are the possible next states?
- What are the probabilities of those transitions?
- What cost or action happens during that transition?

### Step 3: Write the Recurrence
Let $E[X]$ be the expected number of steps from state $X$. Then:
$$E[X] = (immediate \ cost) + \sum_{all \ Y} Pr(X \to Y) \cdot E[Y]$$

### Step 4: Identify Absorbing/Base Cases
These are states where the process ends (e.g., success, match found)
For such states, set:
$$E[\text{absorbing state}] = 0$$
Start solving from these known values.

### Step 5: Optimize Transitions
- Replace summations with prefix/suffix sums when possible.
- Use memoization or DP to avoid redundant computation.
- When probabilities have denominators, compute modular inverses efficiently.
- Avoid brute-force summation when transitions follow an order (like sorted array).

### Step 6: Compute in Correct Order
If the recurrence depends on higher-indexed states (e.g., E[x]), solve in reverse order.
E[x] depends on E[y] for y>x), solve in reverse order.

Otherwise, forward or topological order may work.