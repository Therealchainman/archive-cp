def cellCompete(states, days):
    states = [0] + states + [0] # add two inactive artificial cells around the starting cells
    for _ in range(days):
        nstates = [0 for _ in range(len(states))]
        for i in range(1, len(states)-1):
            nstates[i] = 0 if states[i-1]==states[i+1] else 1
        states = nstates
    return states[1:-1] # remove the two artificial cells 

from typing import List
def getFactors(val: int):
    factors = [] 
    i = 1
    while i*i <= val:
        if val%i==0:
            factors.append(i)
        i+=1
    return factors
def generalizedGCD(num: int, arr: List[int]):
    elem = min(arr) # candidate for GCD
    factors = getFactors(elem) # candidates for GCD,
    for fact in factors[::-1]:
        if all(val%fact==0 for val in arr):
            return fact
    return 1

def generalizedGCD(num, arr):
    def gcd(a, b):
        if a==0:
            return b
        return gcd(b%a, a)
    if num==0:
        return 0
    tgcd = 0
    for elem in arr:
        tgcd = gcd(tgcd,elem)
    return tgcd
