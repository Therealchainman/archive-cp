from collections import Counter
def getRating(index, ratingArr, indicator):
    if len(ratingArr) == 1:
        return int(ratingArr[0], 2)
    nBinaryDigits = len(ratingArr[0])
    if index == nBinaryDigits:
        return -1 # error
    cands = Counter(list(zip(*ratingArr))[index]).most_common()
    interest = str(1+indicator) if len(cands) == 2 and cands[0][1] == cands[1][1] else cands[indicator][0]
    return getRating(index + 1, list(filter(lambda x: x[index] == interest, ratingArr)), indicator)

if __name__ == '__main__':
    with open("inputs/input.txt", "r") as f:
        binaryArr = f.read().splitlines();
        print(getRating(0, binaryArr, 0)*getRating(0, binaryArr, -1))