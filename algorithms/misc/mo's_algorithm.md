# Mo's Algorithm


This is the general outline of the algorithm.  It is brilliant for problems where you need to get answer for many segments or intervals.  It finds the answer for each current window in optimal way.  The only requirement is that the remove and add function need to be relatively fast about O(1) for it to be fast enough. 

Time complexity is roughly $O\big(NF\sqrt{N}\big)$, where F is time complexity of add and remove function

This is when you pick $B = \frac{N}{\sqrt{Q}}$.

```cpp
int block_size;

struct Query {
    int l, r, idx;
    Query(int l, int r, int idx) : l(l), r(r), idx(idx) {}

    bool operator<(const Query &other) const {
        int b1 = l / block_size, b2 = other.l / block_size;
        if (b1 != b2) return b1 < b2;
        if (b1 & 1) return r > other.r;
        return r < other.r;
    }
};

void remove(idx);  // TODO: remove value at idx from data structure
void add(idx);     // TODO: add value at idx from data structure
int getAnswer();  // TODO: extract the current answer of the data structure

vector<int> mo_s_algorithm(vector<Query> queries) {
    block_size = max(1, (int)(N / max(1.0, sqrt(Q))));
    vector<int> answers(Q);
    sort(queries.begin(), queries.end());

    // TODO: initialize data structure
    int curL = 0, curR = -1;
    for (const Query& q : queries) {
        while (curL > q.l) add(--curL);
        while (curR < q.r) add(++curR);
        while (curL < q.l) { remove(curL); ++curL; }
        while (curR > q.r) { remove(curR); --curR; }
        answers[q.idx] = getAnswer();
    }
    return answers;
}
```

Yes, it is exactly zigzagging back and forth! Your intuition is completely correct.If we use a basic sort where the right pointer \(R\) always sorts from smallest to largest (R < other.R), \(R\) will execute a harsh reset action. It will march all the way from the left to the right of the array for one block, and then when the left pointer \(L\) moves to the next block, \(R\) must fly all the way back to the far left to start over. This unnecessary back-and-forth travel wastes performance.To prevent this waste, competitive programmers use an optimization called Parity Sorting (or "zigzag" sorting).

# Mo's Algorithm Time Complexity Summary

The time complexity comes from counting how much the two pointers move overall.

Mo's algorithm keeps one active range:

```text
[curL, curR]
```

For each query, it moves `curL` and `curR` until the current range matches `[l, r]`.

The sorting makes those movements efficient.

## Left Pointer Movement

Queries are grouped by the block of `l`.

If the block size is:

$$
B
$$

then within one block, all left endpoints are close together. So for each query, `curL` moves at most about `B`.

Across `Q` queries, the left pointer contributes:

$$
O(QB)
$$

## Right Pointer Movement

Within each left block, queries are sorted by `r`, so `curR` mostly scans across the array instead of jumping randomly.

There are about:

$$
\frac{N}{B}
$$

left blocks.

For each block, `curR` can move about:

$$
O(N)
$$

So total right pointer movement is:

$$
O\left(\frac{N}{B} \cdot N\right)
=
O\left(\frac{N^2}{B}\right)
$$

## Total Complexity

Adding both pointer costs gives:

$$
O\left(QB + \frac{N^2}{B}\right)
$$

Your template chooses:

$$
B = \frac{N}{\sqrt Q}
$$

Plugging that into the left pointer term:

$$
QB = Q \cdot \frac{N}{\sqrt Q} = N\sqrt Q
$$

Plugging it into the right pointer term:

$$
\frac{N^2}{B}
=
\frac{N^2}{N / \sqrt Q}
=
N\sqrt Q
$$

Therefore, the total complexity is:

$$
O(N\sqrt Q)
$$

When:

$$
Q \approx N
$$

this becomes the common form:

$$
O(N\sqrt N)
$$

## Important Assumption

This complexity assumes that each of these operations is `O(1)`:

```cpp
add(idx);
remove(idx);
getAnswer();
```

If any of those operations are slower, their cost gets multiplied into the pointer movement cost.

## Core Intuition

Mo's algorithm is efficient because it does not recompute each range from scratch.

Instead of paying:

$$
O(r - l + 1)
$$

for every query, it maintains one current window and only pays for the elements that enter or leave the window.

The query ordering makes the total amount of window movement small.