# fraction

## comparison for fractions

Since rounding can cause issues when trying to compare fractions, and comparing is necessary to sort fractions you can use this property to sort fractions

$\frac{x1}{y1} < \frac{x2}{y2}$ => $x_{1} * y_{2} < x_{2} * y_{1}$

```py
# comparator for fractions
class Fraction:
    def __init__(self, num, denom):
        self.num, self.denom = num, denom
    
    def __lt__(self, other):
        return self.num * other.denom < other.num * self.denom
```

## Using GCD to store ratios 

This avoid floating-point form that can have precision issues, and you can perform exact matching between two fractions without the loss of precision. 

```cpp
int g = gcd(nums[p], nums[q]);
pair<int, int> reducedForm = {nums[p] / g, nums[q] / g}; // nums[p] / nums[q] in reduced form
```