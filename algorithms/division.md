# Division

## floor division

Safer to use x // y instead of math.floor(x / y)

## ceil division

safter to use (x + y - 1) // y instead of math.ceil(x / y).  

```py
def ceil(x, y):
    return (x + y - 1) // y 
def floor(x, y):
    return x // y
```