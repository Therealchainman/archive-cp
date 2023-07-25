# Matrix Exponentiation


## Implementation in python

Can be used to solve a system of linear equations with matrix.

AX = B for example, where AX represents matrix multiplication of A and X.

```py
"""
matrix multiplication with modulus
"""
def mat_mul(mat1: List[List[int]], mat2: List[List[int]], mod: int) -> List[List[int]]:
    result_matrix = []
    for i in range(len(mat1)):
        result_matrix.append([0]*len(mat2[0]))
        for j in range(len(mat2[0])):
            for k in range(len(mat1[0])):
                result_matrix[i][j] += (mat1[i][k]*mat2[k][j])%mod
    return result_matrix

"""
matrix exponentiation with modulus
matrix is represented as list of lists in python
"""
def mat_pow(matrix: List[List[int]], power: int, mod: int) -> List[List[int]]:
    if power<=0:
        print('n must be non-negative integer')
        return None
    if power==1:
        return matrix
    if power==2:
        return mat_mul(matrix, matrix, mod)
    t1 = mat_pow(matrix, power//2, mod)
    if power%2 == 0:
        return mat_mul(t1, t1, mod)
    return mat_mul(t1, mat_mul(matrix, t1, mod), mod)

```

Often times you are solving with a transition matrix, which means it allows you to compute the a_n+1 term from the a_n term.

transition_matrix^power*base_matrix = solution_matrix

How can this be applied is from this example. 

This solves a sum of geometrix progression type problem where you want
sum = base^0 + base^1 + base^2 + ... + base^(num_terms-1)

```py
base, num_terms, mod = 3, 4, 7
# exponentiated_matrix*base_matrix = solution_matrix
# exponentiated_matrix = transition_matrix^num_terms
transition_matrix = [[base, 1], [0, 1]]
base_matrix = [[0], [1]]
exponentiated_matrix = mat_pow(transition_matrix, num_terms, mod)
solution_matrix = mat_mul(exponentiated_matrix, base_matrix, mod)
return solution_matrix[0][0]
```