# sys.stdout = open('outputs/output.txt', 'w')
with open("inputs/input.txt", "r") as f:
    A = f.read().splitlines()
    sizeDigitMap = {2,4,3,7}
    outputs = [x[x.find('|')+2:].split() for x in A]
    cnt = sum(1 for out in outputs for pat in out if len(pat) in sizeDigitMap)
    print(cnt)
# sys.stdout.close()