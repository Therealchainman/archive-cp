from collections import *
from functools import *
from itertools import *
import sys
import re
import math
import string
# sys.stdin = open("input.txt", "r")
sys.stdout = open("output.txt", "w")

def main():
    with open('input.txt', 'r') as f:
        data = f.read().splitlines()
        res = 0
        for i, line in enumerate(data):
            _, after_part = line.split(":")
            before, after = after_part.split("|")
            before_numbers = set(map(int, before.split()))
            after_numbers = list(map(int, after.split()))
            cnt = sum(1 for num in after_numbers if num in before_numbers)
            if cnt == 0: continue
            res += pow(2, cnt - 1)
        print(res)             

if __name__ == '__main__':
    main()