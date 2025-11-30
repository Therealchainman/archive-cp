# Mathematics

Some quick mathematical algorithms and formulas.

## Calculating the floor of the square root of an integer

This formulation probably does not work for numbers that are too large in 64 bit, but under 32 bit integers it should be fine.

```cpp
int64 flSqrt(int64 x) {
    int64 y = sqrt(x);
    while (y * y > x) y--;
    while ((y + 1) * (y + 1) <= x) y++;
    return y;
}
```