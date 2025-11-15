# Meta Hacker Cup 2022

# Round 2

## Problem A1: Perfectly Balanced - Chapter 1

prefix sum + static range queries + range sum queries

```py
def main():
    s = input()
    Q = int(input())
    n = len(s)
    unicode = lambda ch: ord(ch) - ord('a')
    psum = [[0] * 26 for _ in range(n + 1)]
    for i in range(n):
        for j in range(26):
            psum[i + 1][j] = psum[i][j]
        psum[i + 1][unicode(s[i])] += 1
    res = 0
    for i in range(Q):
        left, right = map(int, input().split())
        left -= 1
        right -= 1
        if (right - left + 1) % 2 == 0: continue
        mid = (left + right) // 2
        # part 1
        lsum, rsum = [psum[mid + 1][j] - psum[left][j] for j in range(26)], [psum[right + 1][j] - psum[mid + 1][j] for j in range(26)]
        off = sum([abs(lsum[j] - rsum[j]) for j in range(26)])
        if off == 1:
            res += 1
            continue
        # part 2
        lsum, rsum = [psum[mid][j] - psum[left][j] for j in range(26)], [psum[right + 1][j] - psum[mid][j] for j in range(26)]
        off = sum([abs(lsum[j] - rsum[j]) for j in range(26)])
        if off == 1:
            res += 1
    return res

if __name__ == '__main__':
    T = int(input())
    for t in range(1, T + 1):
        print(f"Case #{t}: {main()}")
```

## Problem A2: Perfectly Balanced - Chapter 2

### Solution 1: 

1. online queries
1. decision problem of yes/no if this subarray is an almost perfectly balanced array.
1. R - L + 1 is odd, else it is impossible. 
1. If there are multiple elements with odd frequency in an interval, then it is impossible.
1. You know the value you will be deleting, cause it is only one with odd frequency.
1. hashing that is sensitive to frequency of integers, and not the order.
1. the hash is just to map each integer from 10^6 to some random integer in the range 1, 10^18, or a 64 bit integer. 
1. just let the left half always be one larger than the right, and then take difference of both hashes, and map those two values back to an integer, they must match to the same integer, and then that is the extra integer on left half, then looking in a set determine if this integer is ever within that interval. 

```cpp

```

## Problem B: Balance Sheet

### Solution 1: DAG, K longest paths in a DAG, merge sorted lists, sorting, dynamic programming

1. K longest paths in a DAG
1. implicit DAG, don't actually construct the edges, cause that would be O(N^2) work.
1. Instead there is a clever merging to retain the K best paths.
1. given a particular day x, you have a set of seller nodes and buyer nodes. 
1. sorted list up to size K for each seller node
1. add zero for each seller to there sorted list, to represent that they are going to start a path, that is they will be the first seller in the path.
1. sort the seller nodes by selling price ascending and same for buyer nodes for day x. 
1. Then keep a sorted list that is descending that has the top k best paths, but these are calculated based on the profit of a selling path minus the selling price Y_j.
1. Then for a buyer you take that descending list and just have to add the buying price X_i to all of them.
1. Then you will need to merge other seller lists of size k into that sorted list that has the current k best paths.  But that should only take O(K) time since both lists are size K.
1. So the time complexity is you have N seller and N buyer nodes, so that is O(N) nodes, and for each node you are doing O(K) work to merge the lists, so O(N*K) per day.
1. Take the K best paths over all buyers sorted list for all days. 

```cpp

```

## Problem C: Balance Scale

combinatorics, probability, math, factorials, multiplicative mod inverse, conditional probability, uniform distribution

```py
mod = int(1e9 + 7)
TOTAL = 3_000 * 3_000

def mod_inverse(v):
    return pow(v, mod - 2, mod)

def factorials(n):
    fact, inv_fact = [1] * (n + 1), [0] * (n + 1)
    for i in range(2, n + 1):
        fact[i] = (fact[i - 1] * i) % mod
    inv_fact[-1] = mod_inverse(fact[-1])
    for i in reversed(range(n)):
        inv_fact[i] = (inv_fact[i + 1] * (i + 1)) % mod
    return fact, inv_fact

def main():
    N, K = map(int, input().split())
    cookies = [None] * N
    for i in range(N):
        c, w = map(int, input().split())
        cookies[i] = (c, w)
    C_1, W_1 = cookies[0]
    C_less = C_equal = C_greater = 0
    for i in range(1, N):
        c, w = cookies[i]
        if w < W_1:
            C_less += c
        elif w  == W_1:
            C_equal += c
        else:
            C_greater += c
    M = C_1 + C_less + C_equal + C_greater
    def nCr(n, r):
        return (fact[n] * inv_fact[r] * inv_fact[n - r]) % mod if n >= r else 0
    comb_neutral = nCr(M - C_greater, K + 1)
    comb_less = nCr(C_less, K + 1)
    comb_equal = comb_neutral - comb_less
    comb_total = nCr(M, K + 1)
    # P(AUB) = P(A|B) * P(B), B = equal, A = equal cookie from batch 1 is the tie breaker
    prob_equal = (comb_equal * mod_inverse(comb_total)) % mod
    conditional_prob = (C_1 * mod_inverse(C_1 + C_equal)) % mod
    res = (prob_equal * conditional_prob) % mod
    return res

if __name__ == '__main__':
    fact, inv_fact = factorials(TOTAL)
    T = int(input())
    for t in range(1, T + 1):
        print(f"Case #{t}: {main()}")
```

## Problem D1: Work Life Balance - Chapter 1

### Solution 1:

1. fenwick tree for range queries to get the counts of 1,2, 3s for both subarrays
1. take the sum of both subarrays and say one sum is less than other L < H
1. There are only the following swaps that are useful swap 1, 3, decrease by 4, swap 1, 2, or 2, 3 decrease by 2, or swap 2, 1 or 3, 2 increase by 2. 
1. basically you'd want to perform decrease by 4 as much as you can until you run out of 1 in L, and 3 in H.  Then decrease by 2. or decrease by 4 until the difference is -2, then increase by 2 once. 

```py

```

## Problem D2: Work Life Balance - Chapter 2

### Solution 1:

1. query how many 1s are within an interval [0, R]. 
1. binary search so that you move to the largest L, such that the count of 1s in interval [l, R] is at least x.
1. have another fenwick tree for calculating the sum of the interval [L, R].
1. Another fenwick tree that has a value for index if there is a 1 at that index, so I can calculate for an interval the sum of all index for 1s for instance.
1. 

```cpp

```

# Round 3

## Problem A: Fourth Player

### Solution 1: greedy, game theory

1. The idea is to just play all the cards for the players in order. A1, B1, A2, B2. 
1. And sort in descending order A1 cards, and say that is how player A1 will play them, doesn't really matter.  (just will make later process easier)
1. Then what will player B1 do? You would try to cover A1 largest cards, because that would tip the point in your favor. So just go through your cards in descending order, and use a two pointer, and try to cover as many A1 cards as you can.  Where trying to cover means you play the card that is larger than what A1 played at that turn.  Any cards you can't cover just play your remaining smallest valued cards, in a sense you just throwing them away. But you've played your largest cards in a way so maybe you will get a point for team B.
1. What will player A2 do? First thing is try to cover where team B is currently winning in sorted order just like you did with player B1 above, and now you can tip the point back to team A, that is going to be helpful.  Now with the remaining cards A2 has, you should play your largest remaining on the smallest A1 cards that are winning, that way you make it harder for that to lose to B2. 
1. What will player B2 do? Just try to cover as many places where team A is winning.  So take your largest and try to cover the index where B is currently leading and it is the largest card value just less than this one.  Two pointers
1. At the end you know how many points team A scores, based on where team A still has largest card played.

```cpp

```

## Problem B: Third Trie

### Solution 1: combinatorics, trie, dfs

1. count total number of triplets of the N tries.
1. Create a mega trie of all the tries, which tracks count for each node in the trie, that represents the number of tries that contained that string prefix.
1. Basically for each string prefix you can calculate the number of triplets of tries that do not contain that prefix, by subtracting the number of tries that contain it N - k, and choose 3 from that. 
1. Then that would give you the number of triplets that do contain that prefix. 

```cpp
int N, M;

int64 choose3(int64 n) {
    return n * (n - 1) * (n - 2) / 6;
}

struct Node {
    int children[26];
    bool isLeaf;
    int64 cnt;
    void init() {
        memset(children, 0, sizeof(children));
        isLeaf = false;
        cnt = 0;
    }
};
struct Trie {
    vector<Node> trie;
    void init() {
        Node root;
        root.init();
        trie.emplace_back(root);
    }
    void insert(const vector<vector<pair<int, char>>>& adj, int u = 0, int cur = 0) {
        trie[cur].cnt++;
        for (const auto& [v, c] : adj[u]) {
            int idx = c - 'a';
            if (trie[cur].children[idx] == 0) {
                Node node;
                node.init();
                trie[cur].children[idx] = trie.size();
                trie.emplace_back(node);
            }
            insert(adj, v, trie[cur].children[idx]);
        }
    }
    int64 dfs(int cur) {
        int64 ans = choose3(N) - choose3(N - trie[cur].cnt);
        for (int i = 0; i < 26; ++i) {
            if (trie[cur].children[i]) {
                ans += dfs(trie[cur].children[i]);
            }
        }
        return ans;
    }
};
```

## Problem C: Second Mistake

### Solution 1: rolling hash algorithm

1. Rolling hash algorithm is a way to represent a string as a single integer value, that is sensitive to the order of characters in the string.
1. Create a hash for each string in the vocabulary, where you create a hash for if one character is wrong at each index.  In total that means you will only have hashed 3L strings, so that is manageable.
1. It is important that you store these hashes in the following sums, so you have the sum of the hashes, and then you also want a sum of the hashes at a specific index, hsum[h], hsum_idx[h][i]. 
1. To generate the hashes is in O(1) time is not that difficult if you just take the hash of the entire string, then subtract the hash of the prefix substring ending at index i. Then adding back the hash of the prefix substring ending at index i - 1, then adding the hash of the character you are replacing at index i. 
1. Now you go through the queries, and generate the hash for all the possible mistakes at each index, so generate 3L hashes for each query string.
1. And you calculate the hash for if you replaced character at index i with character ch, suppose you replace 'e' with 'm'.  Now you want to find how many hashes match this, because that would mean the character is wrong at the index i, but also the hashes are storing those for where the character at another index is also wrong. 
1. So this makes you have two indexes that are wrong!!
1. So you want the hsum[h] - hsum_idx[h][i] to get the sum of all hashes that match this hash, but not at index i. But also you divide by 2, because you have double counted the cases where both indexes are wrong.

```cpp

```

## Problem D1: First Time - Chapter 1

### Solution 1: 

1. Once a bin is all the same color it will bel ike that any time later as well.
1. disjoint set union data structure, I thought I could just binary search and construct the state of the disjoint set union at that time. and that it would be FFFFTTTTT, and find the first time it becomes T, where all the bins contain same color elements. 
1. And I thought uniting two is relatively easy, just track the current color and what representative/root node it points to, then you can know how to merge and so on.  Why would this not work? 
1. So this should actually work. 

```cpp

```

## Problem D2: First Time - Chapter 2

### Solution 1: union find, sparse table range max query, queries attached to components, small-to-large merging

1. Take the fence at some time t and write down the DSU representative for each position 1..N. Now scan left to right and group consecutive positions that have the same representative into maximal blocks. Those blocks are the segments S1, S2, …, SG, and their lengths add up to N.
1. “Maximal” means you cannot extend a block one step left or right without hitting a different representative.
1. G is simply the number of such blocks, also called runs.
Quick example with N = 10:
Suppose the reps along the fence are
[1, 2, 2, 2, 5, 6, 6, 8, 9, 10].
The maximal runs are
[1] [2,2,2] [5] [6,6] [8] [9] [10].
So G = 7 and the segment lengths are [1, 3, 1, 2, 1, 1, 1], which sum to 10.
1. Case 1: If N is divisible by K, We know that the colors are the same in each bin if the length of each segment is divisible by K, or in other words the gcd(S1, S2, ..., S_G) is divisible by K. 
1. Case 2: If N is not divisible by K, We'll have a left rightmost remainder of some length less than K., denote it's letngth as R, if R < size of last segment S_G, then it is impossible.  If it is greater than split up that last segment and basically take L = S_G - R, and now gcd(S1, S2, ..., S_G-1, L) is divisible by K is the check.

1. Treat each repaint as merging color classes.
1. For every adjacent pair (i, i+1), record the first time t when they land in the same class using the DSU + query buckets.
1. A block of length K is monochromatic iff all its adjacent pairs have already matched, so the time for a block is the max of those pair-times inside it.
1. Precompute a sparse table over those pair-times, then for each K, scan blocks and take block maxima quickly, summing answers over K.

1. This is a classic offline dynamic connectivity under only unions, solved with union by size and query buckets.
1. What are the query buckets? 
query buckets (attach to components trick)
Each pending "when do u and v meet?" query is attached to the current components containing u and v. 

1. id[u] = which live component carries color name u.  Repaints are phrased in color names, so we need this label -> component mapping.
1. ans[i] = earliest time neighbors i and i + 1 coincide.
1. Q[u] = list of queries attached to component u.
1. a[u] = members of component u.
1. col[u] = current component id of element at index u.
1. On the exact merge where the endpoints’ components meet, we will scan only the smaller side. If the pair were attached to only one endpoint, it might be attached to the larger side and you would miss it. Putting it in both guarantees the id is present in the small side and will be seen.

hint at solution:

```cpp
const int N = 600600;
const int LOG = 20;
int id[N];
int col[N];
vector<int> a[N];
vector<int> Q[N];
int ans[N];
int n;
int p2[N];
int sparse[LOG][N];

void unite(int v, int u, int t) {
	v = col[v];
	u = col[u];
	if (v == u) return;
	if ((int)a[v].size() < (int)a[u].size()) swap(v, u);
	for (int z : Q[u]) {
		int p = z / 2;
		int q = p + 1;
		if ((col[p] == v && col[q] == u) || (col[p] == u && col[q] == v)) {
			ans[p] = t;
		} else {
			Q[v].push_back(z);
		}
	}
	for (int w : a[u]) {
		col[w] = v;
		a[v].push_back(w);
	}
}

int getMax(int l, int r) {
	if (l >= r) return 0;
	int k = p2[r - l];
	return max(sparse[k][l], sparse[k][r - (1 << k)]);
}
int query(int k) {
	int res = 0;
	for (int l = 0; l < n; l += k) {
		int r = min(n, l + k);
		res = max(res, getMax(l, r - 1));
	}
	if (res == N) return -1;
	return res;
}

void solve() {
	int m;
	scanf("%d%d", &n, &m);
	eprintf("n = %d, m = %d\n", n, m);
	for (int i = 0; i < n; i++)
		ans[i] = N;
	for (int i = 0; i < n; i++) {
		a[i].clear();
		Q[i].clear();
		id[i] = i;
		col[i] = i;
		a[i].push_back(i);
	}
	for (int i = 1; i < n; i++) {
		Q[i].push_back(2 * i - 1);
		Q[i - 1].push_back(2 * i - 1);
	}
	for (int t = 1; t <= m; t++) {
		int v, u;
		scanf("%d%d", &v, &u);
		v--;u--;
		if (id[v] == -1) continue;
		if (id[u] == -1) {
			id[u] = id[v];
			id[v] = -1;
			continue;
		}
		int z = id[v];
		id[v] = -1;
		unite(z, id[u], t);
	}
	for (int i = 0; i < n; i++)
		sparse[0][i] = ans[i];
	for (int t = 0; t < LOG - 1; t++)
		for (int i = 0; i + (1 << (t + 1)) <= n; i++)
			sparse[t + 1][i] = max(sparse[t][i], sparse[t][i + (1 << t)]);
	ll ans = 0;
	for (int k = 1; k <= n; k++)
		ans += query(k);
	printf("%lld\n", ans);
}
```

## Problem E1: Zero Crossings - Chapter 1

### Solution 1: 

1. One of the simplifying factors for this is that you can solve it with offline query algorithm because each marble position is independent of the other marbles.
1. I think the key is given two points A and B.  If they are contained within polygon C and polygon D, if C = D, it is trivial and you can swap them. If C != D, then you require that polygons C and D are enclosed by the same polygons.  

1. Two marbles can be swapped iff the region of the plane that holds the red marble is “equivalent” to the region that holds the blue marble, where equivalence means:
- the same nesting pattern of rubber bands inside it (bands are unlabeled and may permute), and
- the constraints “no touching/crossing, end state matches the initial set of polygonal outlines” stay satisfiable.
1. builds the nesting tree of polygons (who’s inside whom),
1. assigns a canonical type to each region/subtree (so two regions are equal if their subtrees are isomorphic as unordered trees),
1. locates, for each query point, the region node that contains it, and
1. answers YES if the two region nodes have the same canonical type.

1. The tree that is built is like this. 
1. Node 0 the unbounded outside region.
1. For each polygon Pi, create a node “inside of Pi”.
1. Put an edge from the region that directly contains Pi to the node “inside of Pi”.
1. So every edge corresponds to crossing one polygon boundary; nesting becomes parent→child.

1. Each marble sits in exactly one region node of this tree: call them R and B.
1. Two region nodes are equivalent iff their rooted subtrees are isomorphic as unordered trees: you can pair each child region of R with a child region of B so that the paired children themselves have isomorphic subtrees, recursively, and order among siblings doesn’t matter.

1. id(u) = The id is a canonical label for the isomorphism class of a rooted, unordered subtree.
- Two nodes get the same id exactly when their subtrees have the same shape ignoring child order and labels.
- So id does not identify a node, it identifies the subtree structure hanging from that node.

1. Compute a polynomial rolling hash of the shape identifier for each subtree, from the root to node u.
1. Order matters, if two nodes have the same ancestor chain (in the unabeled sense) but different own ids, they get different h. If anything along the chain differs, the polynomial value changes.

1. Why you have to rotate the points, the sweep keeps an ordered set of “active” edges by their y-value at the current x. A vertical edge breaks that idea.  The problem is you can't divide by zero, which can happen with vertical line. 

1. Fix an x value. Draw the vertical line x = X. Every polygon edge that spans this x intersects that vertical line at exactly one y value.
**The scenarios of the containment of a point or start of a polygon**.
1. Both neighbors belong to the same polygon k
- Picture a single convex polygon crossing the vertical line twice. The section of the line between its two crossings is the interior of that polygon. The strip where your point lies is inside that one polygon and is bordered above and below by the same polygon’s edges. So the region is the node “inside polygon k”.
1. Neighbors belong to different polygons and one is the direct parent of the other
- Suppose the lower edge is from polygon u and the upper edge is from polygon v, and u directly contains v (no other polygon between them). On the vertical line, intersections alternate in and out as you pass edges. The strip between u and v is therefore inside u but outside v. So the region containing P is the parent polygon’s interior region.
1. Neighbors are different and neither is parent of the other
- Then they must be siblings under a common ancestor. Since polygons do not cross, there is some highest polygon w that contains both u and v, and neither u nor v contains the other. On the vertical line, you see crossings that look like: enter w, enter some children, leave some children, leave w. The strip between two siblings lies in the region “inside their common parent w but outside both children.” So the region of the strip is the parent region of both, which is the same value par[u] == par[v].

1. Because polygons never cross or touch, that strip belongs to exactly one region, anything you query. 
1. horizontal ray because the slope is 0, and you just have the y value. 

1. Take concentric loops or nested polygons. Spin the page. Which loop is inside which hasn’t changed—you only changed your viewpoint.


why the leftmost point of polygon is important?
```cpp

```

## Problem E2: Zero Crossings - Chapter 2

### Solution 1: 

1. You can still construct the nesting tree of polygons (who’s inside whom), with line sweep algorithm.
1. But you can not determine where a point is contained in the polygon offline, you have to do it online.
1. 

```cpp

```

