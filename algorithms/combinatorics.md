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

