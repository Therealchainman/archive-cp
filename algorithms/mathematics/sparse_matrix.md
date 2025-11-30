# SPARSE MATRIX ALGORITHMS

Matrices that mostly contain zeroes are said to be sparse.

The scipy.sparse module provides data structures for 2D sparse matrices. There are seven available sparse matrix formats:

csc_matrix: Compressed Sparse Column

csr_matrix: Compressed Sparse Row

bsr_matrix: Block Sparse Row

lil_matrix: List of Lists

dok_matrix: Dictionary of Keys

coo_matrix: Coordinate

dia_matrix: Diagonal

Each sparse format has certain advantages and disadvantages. For instance, adding new non-zero entries to a lil_matrix is fast, however changing the sparsity pattern of a csr_matrix requires a significant amount of work. On the other hand, operations such as matrix-vector multiplication and matrix-matrix arithmetic are much faster with csr_matrix than lil_matrix. A good strategy is to construct matrices using one format and then convert them to another that is better suited for efficient computation.


sparsity = count zero elements / total elements

## Compressed Sparse Row (CSR) Matrix

```py
class CompressedSparseRowMatrix:
    def __init__(self, matrix):
        R, C = len(matrix), len(matrix[0])
        self.values, self.col_indices, self.row_indices = [], [], [0]
        for r in range(R):
            for c in range(C):
                if matrix[r][c] == 0: continue
                self.values.append(matrix[r][c])
                self.col_indices.append(c)
            self.row_indices.append(len(self.values))
```
## Compressed Sparse Column (CSC) Matrix

```py
class CompressedSparseColumnMatrix:
    def __init__(self, matrix):
        R, C = len(matrix), len(matrix[0])
        self.values, self.col_indices, self.row_indices = [], [0], []
        for c in range(C):
            for r in range(R):
                if matrix[r][c] == 0: continue
                self.values.append(matrix[r][c])
                self.row_indices.append(r)
            self.col_indices.append(len(self.values))
```