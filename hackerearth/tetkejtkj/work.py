def squares():
    for i in range(100000):
        yield i*i
arr = []
for i in squares():
    arr.append(i)
    if i>100000000000:
        break

print(arr)


"""
There is only 100k elements in the array.

565

"""