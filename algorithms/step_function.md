# Step Function

## Parameterized step function

This is a simple function that will generate step function outputs, with given step size and height.  But also allows for horizontal and vertical shifts or translations of the data.

```py
def floor(x, y):
    return x // y
# provide step_size >= 1, and height >= 1
def step_function(x, step_size, height, horizontal_shift = 0, vertical_shift = 0):
    return floor(x + horizontal_shift, step_size) * height + vertical_shift
```