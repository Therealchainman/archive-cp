# Combinatorics

Combinatorics is the branch of mathematics studying the enumeration, combination, and permutation of sets of elements and the mathematical relations that characterize their properties. Mathematicians sometimes use the term "combinatorics" to refer to a larger subset of discrete mathematics that includes graph theory.

## Combinations

A combination is a selection of items from a collection, such that (unlike permutations) the order of selection does not matter.

- order does not matter

This is an implementation in C++ that will return an array of all combinations of size K.  It can run in binomial coefficient time complexity that is O(k $\binom{n}{k}$)

```cpp
vector<int> A, cur;
vector<vector<int>> combinations;

void dfs(int i) {
    if (cur.size() == K) {
        combinations.push_back(cur);
        return;
    }
    for (int j = i; j < N; j++) {
        cur.emplace_back(A[j]);
        dfs(j + 1);
        cur.pop_back();
    }
}
```

## Permutations

A permutation is an arrangement of items in a specific order. For example, the numbers 1, 2, and 3 can be arranged in six different ways: 123, 132, 213, 231, 312, and 321. These are all permutations of the set {1, 2, 3}.

- order matters

## Derangements

The number of derangements of $n$ numbers, expressed as $!n$, is the number of permutations such that no element appears in its original position. Informally, it is the number of ways $n$ hats can be returned to $n$ people such that no person recieves their own hat.

## Stars and Bars Method

The stars and bars method is a popular combinatorial technique used to solve problems of distributing indistinguishable objects (stars) into distinguishable boxes (bars).

### Problem Statement

Given $n$ indistinguishable stars and $k$ distinguishable boxes, the problem is to find the number of ways to distribute the stars into the boxes.

### Formula

The number of ways to distribute $n$ indistinguishable stars into $k$ distinguishable boxes is given by the formula:

$$
\binom{n+k-1}{k-1}
$$

### Explanation

1. **Stars**: Represent the indistinguishable objects we want to distribute.
2. **Bars**: Represent the dividers between different boxes. If we have $k$ boxes, we need $k-1$ bars to create the divisions.
3. **Total Objects**: The total number of objects (stars + bars) is $n + (k - 1) = n + k - 1$.
4. **Choosing Positions**: We need to choose $k - 1$ positions for the bars out of the total $n + k - 1$ positions.

### Example

Suppose we want to distribute 5 indistinguishable stars into 3 distinguishable boxes. We can represent the stars and bars as follows:

```
*****|**|*
```

In this case, we have 5 stars and 2 bars. The number of ways to arrange these is given by:

$$
\binom{5+3-1}{3-1} = \binom{7}{2} = 21
$$

Thus, there are 21 ways to distribute 5 indistinguishable stars into 3 distinguishable boxes.

## Other things to know about Stars and Bars method

Indistinguishable things: the K stars. Each star represents one bin, and we treat bins as unlabeled because we store states canonically, for example as a sorted tuple or a histogram. Swapping two bins with the same or different loads is the same state, so the stars are indistinguishable.

Distinguishable things: the T+1 buckets labeled by load values 0,1,...,T where T = N/K. These buckets are different because “load 0” is not the same as “load 1”, etc. In stars and bars terms the bars are identical separators, but their positions determine how many stars fall in each labeled bucket.

## Stars and Bars method and Fibonacci sequence

If you are counting the number of binary strings with non-adjacent ones, you can compute with Fibonacci sequence.
$n = 0,1,2,3,4,5:\quad a_n = 1,2,3,5,8,13 \;=\; F_2, F_3, F_4, F_5, F_6, F_7$

But this can also be calculated with stars and bars method, thus creating a connection between these two concepts.

Binomial formula for Fibonacci numbers.  Let's fix n and count strings with exaclty k ones (no two adjacent). Put the k ones down first; to keep them non-adjacnet you must place at least one zero between consecutive ones.  That uses up k - 1 zeros.

extra zeros = $n - k - (k - 1) = n - 2k + 1$
These are to be distributed into the k + 1 gaps, distributing the indistinguishable items into k + 1 boxes is classic stars and bars.

$\binom{n - k + 1}{k}$
valid for $ 0 \leq k \leq \left\lfloor \frac{n+1}{2} \right/rfloor$
Summing over all feasible k,
$$a_n = \sum_{k=0}^{\left\lfloor (n + 1) / 2 \right\rfloor} \binom{n - k + 1}{k}$$
This sums satisfies Fibonacci recurrence $a_n=a_{n-1} + a_{n-2} and $a_n=F_{n+2}$


## Using combinatorics for calculating the sum of the lengths of all subarrays

Start from the sum of lengths of all subarrays. That is a known formula
sum of lengths = C(N + 2, 3).
Why: imagine N+2 markers 0, 1, ..., N, N+1. Pick any triple L < M < R.
That triple means subarray (L+1 .. R-1) and a chosen element M inside it.
Each subarray of length m gets counted exactly m times. So the total count of triples is the sum of all subarray lengths.

```cpp
int64 choose3(int64 n) {
    return n * (n - 1) * (n - 2) / 6;
}
```

## Vandermonde's Identity

Vandermonde's identity states that for any non-negative integers m, n, and r, the following equation holds:
$$
\sum_{k=0}^{r} \binom{m}{k} \binom{n}{r-k} = \binom{m+n}{r}
$$
or reformulated with change of variables:
$$
\sum_{k_1 + k_2 = r} \binom{m}{k_1} \binom{n}{k_2} = \binom{m+n}{r}
$$

This means the product of binomial coefficients with the constraint the bottom part or elements being chosen sum to r is equal to the binomial coefficient of the sum of the top parts choosing r elements.

good way to look at it is you come up with a single binomial coefficient on the right side that is the sum of the top and the bottom in the product. 


