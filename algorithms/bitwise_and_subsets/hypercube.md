## 🧊 Hypercube and Bipartite Graph Explanation

### 🔷 What is an **n-dimensional hypercube**?

An **n-dimensional hypercube** (also called an **n-cube**, denoted \( Q_n \)) is a graph defined as:

- **Vertices**: All binary strings of length \( n \) (i.e., numbers from 0 to \( 2^n - 1 \) in binary).
- **Edges**: Two vertices are connected if and only if their binary representations differ in **exactly one bit**.

Examples:

- \( Q_1 \): `0` and `1` — a single edge.
- \( Q_2 \): `00`, `01`, `10`, `11` — a square.
- \( Q_3 \): Eight vertices — forms a cube.

Every new dimension doubles the number of vertices and connects each vertex to a new one that differs in one additional bit.

---

### 🟦 Why is the hypercube a **bipartite graph**?

A graph is **bipartite** if its vertex set can be split into two disjoint sets \( A \) and \( B \) such that:

- No two vertices within the same set are connected.
- Every edge connects a vertex from \( A \) to one in \( B \).

#### ✅ In the hypercube:

We partition vertices based on the **parity** of the number of 1's in the binary string:

- **Set A**: Strings with an **even** number of 1’s.
- **Set B**: Strings with an **odd** number of 1’s.

Now consider any edge — it connects two strings that differ by exactly one bit. Flipping one bit always changes the parity (even ↔ odd), so:

> Every edge connects Set A to Set B.

➡️ This proves that the hypercube \( Q_n \) is bipartite.

---

### 🧠 Why does this matter for our problem?

We are asked to find the **largest possible set of binary strings of length \( n \)** such that **no two differ in exactly one bit**.

In graph terms:

- We want the largest **independent set** in \( Q_n \) — a set of vertices with no edges between them.
- Since \( Q_n \) is bipartite, its largest independent set is exactly one of the partitions.
- Both partitions (even parity and odd parity) have exactly \( 2^{n-1} \) elements.

Therefore:

> Using all binary strings with an **even number of 1’s** gives us a **maximum-size valid set**.

---

### 🔚 Summary

| Concept | Description |
|--------|-------------|
| **Hypercube \( Q_n \)** | Graph with \( 2^n \) binary strings of length \( n \) as vertices |
| **Edge Rule** | Two nodes are connected if they differ in exactly one bit |
