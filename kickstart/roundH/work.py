
def create(num, N):
    """
    Create a list of numbers from 0 to num
    """
    A = num
    while len(num)<N:
        for x in A:
            num += x
    return num

if __name__ == '__main__':
    num = "98765432101"
    T, N, n = 100, 500000, 100
    res = create(num, N)
    with open("input.txt", "w") as f:
        f.write("1\n")
        res = create(num,50)
        f.write(f"{len(res)}\n")
        f.write(res)
        # f.write(f"{T}\n")
        # for i in range(max(10,T)):
        #     f.write(f"{len(res)}\n")
        #     f.write(f"{res}\n")
        # res = create(num, n)
        # for i in range(max(0,T-10)):
        #     f.write(f"{len(res)}\n")
        #     f.write(f"{res}\n")

