from collections import *
from functools import *
from itertools import *
import time

values = {'=': -2, '-': -1, '0': 0, '1': 1, '2': 2}
valuesToSnafu = {v: k for k, v in values.items()}
base = 5
cache = {}


def snafu_to_decimal(snafu):
    res = 0
    for v in map(lambda num: values[num], snafu):
        res = (res*base + v)
    return res

def decimal_to_snafu(num):
    res = []
    while num > 0:
        if num % base == 3:
            res.append('=')
            num += 2
        elif num % base == 4:
            res.append('-')
            num += 1
        else:
            res.append(valuesToSnafu[num % base])
        num //= base
    return ''.join(res[::-1])

def main():
    with open('input.txt', 'r') as f:
        data = f.read().splitlines()
        total = 0
        for snafNum in data:
            num = snafu_to_decimal(snafNum)
            print(f'snafu: {snafNum} decimal: {num}')
            total += num
        return decimal_to_snafu(total)

if __name__ == '__main__':
    start_time = time.perf_counter()
    print(main())
    end_time = time.perf_counter()
    print(f'Time Elapsed: {end_time - start_time} seconds')