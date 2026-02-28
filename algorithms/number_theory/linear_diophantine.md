# Linear Diophantine Equations

Equation of the form $Ax + By = N$ where $A$, $B$, and $N$ are given integers, and we want to find integer solutions for $x$ and $y$.

## Solving special case of Linear Diophantine Equations

This is method to find one solution or to count the number of possible solutions. 

This is the general algorithm for that process.
given ax + by = c, and to find positive integer solutions, x > 0 and y > 0, you can do the following steps.


1. calculate g = gcd(a, b)
2. if c % g != 0, then there are no solutions
3. calculate a' = a / g, b' = b / g, c' = c / g
4. solve a'x = c' (mod b') using multiplicative inverse, so you just need x0 = c' * (a')^-1 (mod b'), this will give you the smallest positive solution for x.
5. then you require that c - ax >= b, which gives x <= (c - b) / a, so x >= 1 and x <= (c - b) / a, so that is your X_max = (c - b) / a.
6. The number of positive integer solutions is if x0 > X_max, then it is 0. otherwise it is 1 + (X_max - x0) / b'.

Why does the multiplicative inverse exist in step 4? Because a' and b' are coprime, since we divided by their gcd. Therefore, there exists an integer k such that a'k ≡ 1 (mod b'), which means that the multiplicative inverse of a' modulo b' exists.

when x0 = 0 from inverse, you need to set x0 = b' because we want positive integer solutions, and the smallest positive solution for x would be b' in that case.