"""
Get all the factors (prime and non-prime) for an integer num
"""
from math import sqrt
def divisors(num):
    div_arr = []
    for i in range(1, int(sqrt(num))+1):
        if num%i==0:
            div_arr.append(i)
            div_arr.append(num//i)
    return list(set(div_arr))