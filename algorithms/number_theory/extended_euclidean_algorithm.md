# Extended Euclidean Algorithm

## Solving Diophantine equaion with extended euclidean algorithm

Assume that this is not the case a = b = 0

Example of how to implement it to solve the equation ax + by = c
if a or b or both are negative, you just need to include -a, -b into the function.  

```cpp
int64 extendedEuclidean(int64 a, int64 b, int64 &x, int64 &y) {
    if (b == 0) {
        x = (a >= 0 ? 1 : -1);
        y = 0;
        return llabs(a);
    }
    int64 x1, y1;
    int64 g = extendedEuclidean(b, a % b, x1, y1);
    x = y1;
    y = x1 - y1 * (a / b);
    return g;
}

// Solve a*x + b*y = c
// Returns true if a solution exists and writes one solution into x, y.
// General solution: x = x0 + t*(b/g), y = y0 - t*(a/g), for any integer t.
bool linearDiophantine(int64 a, int64 b, int64 c, int64 &x, int64 &y, int64 &g) {
    int64 x0, y0;
    g = extendedEuclidean(a, b, x0, y0);
    if (c % g != 0) return false;
    int64 k = c / g;
    x = x0 * k;
    y = y0 * k;
    return true;
}
```


By the Extended Euclidean Algorithm, an integer solution $(x_0, y_0)$ exists precisely when $\gcd(A,B)$ divides $N$.  
Let $g = \gcd(A,B)$. If this condition holds, all integer solutions are given by  

$x = x_0 + k \cdot \frac{B}{g}$  
$y = y_0 - k \cdot \frac{A}{g}$  

for any integer $k \in \mathbb{Z}$.

Define  
$d_x = \frac{B}{g}$, $d_y = \frac{A}{g}$.  

Then the parametric solution can be written as  

$x(k) = x_0 + k \, d_x$  
$y(k) = y_0 - k \, d_y$.



### Example of how to use the linear diophantine function to solve a problem

In this problem you want to find the solution with smallest non-negative x and also the solution with smallest non-negative y, while keeping the other variable non-negative as well.

Just note that A > 0 and B > 0 and C > 0 in this problem.

solve for the y(t) >= 0 and x(t) >= 0 to find tmin and tmax.  Do the algebra given x(t) = x0 + t*dx and y(t) = y0 - t*dy

```cpp
int64 A, B, C;

// A > 0 and B > 0 and C > 0
void solve() {
    cin >> A >> B >> C;
    // I want to solve for x, y: A*x + B*y = C
    // but I want to find the solution with the smallest non-negative x
    // and also find another solution with the smallest non-negative y.
    int64 x0, y0, g;
    if (!linearDiophantine(A, B, C, x0, y0, g)) {
        cout << -1 << endl;
        return;
    }
    int64 dx = B / g;
    int64 dy = A / g;
    // x = x0 + t*dx
    // y = y0 - t*dy
    int64 tmin = - x0 / dx;
    int64 tmax = y0 / dy;
    if (tmin > tmax) {
        cout << -1 << endl;
        return;
    }
    int64 x1 = x0 + tmin * dx, y1 = y0 - tmin * dy;
    int64 x2 = x0 + tmax * dx, y2 = y0 - tmax * dy;
    int64 ans = min(x1 + 3LL * y1, x2 + 3LL * y2);
    cout << ans << endl;
}
```