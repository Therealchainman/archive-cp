# Transformations

Matrix transformations

These following are using extra memory for the rotations.  They are not performing in-place transformations.

## vertical flip

invert the row order

```py
for r, c in product(range(R), range(C)):
    mat[r][c] = st[R - r - 1][c]
```

## horizontal flip

invert the column order

```py
for r, c in product(range(R), range(C)):
    mat[r][c] = st[r][C - c - 1]
```

## horizontal + vertical flip = 180 degrees rotation

invert the row and column order

```py
for r, c in product(range(R), range(C)):
    mat[r][c] = st[R - r - 1][C - c - 1]
```

