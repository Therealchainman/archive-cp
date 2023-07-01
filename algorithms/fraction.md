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