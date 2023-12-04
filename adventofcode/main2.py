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
        cards = [1] * len(data)         
        for card_index, line in enumerate(data):
            _, after_part = line.split(":")
            before, after = after_part.split("|")
            before_numbers = set(map(int, before.split()))
            after_numbers = list(map(int, after.split()))
            cnt = sum(1 for num in after_numbers if num in before_numbers)
            for i in range(card_index + 1, card_index + cnt + 1):
                cards[i] += cards[card_index]
        print(sum(cards))             

if __name__ == '__main__':
    main()