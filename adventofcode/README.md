


# Reading inputs in python

I like to use the following to read in input from an input.txt

```py
with open("inputs/input.txt", "r") as f
```
This reads it in as text format. 

If you are given an input that is a grid of integers such as 

111222
232322
234242

This is a clean way to create a 2d array of integers.
```py
data = []
lines = f.read().splitlines()
for line in lines:
    data.append([int(x) for x in line])
```

If you are given an input that is a grid of characters, and don't want to convert to integers

```py
data = f.read().splitlines()
```

# Debugging tricks

## If output text too large for terminal, write it to a file. 

This is a method to write all print statements in the python script to a file before you close it. 
it sets the system stdout to be piped to this file.  
```py
sys.stdout = open('outputs/output.txt', 'w')
sys.stdout.close()
```