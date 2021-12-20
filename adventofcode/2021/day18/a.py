from dataclasses import dataclass

class Pair:
    def __init__(self, left, right):
        self.left = left
        self.right = right
    def __repr__(self):
        return f'[Pair {self.left}, {self.right}]'
  
class SnailNumber:
    def __init__(self):
        self.numbers = self.data_loader()

    def data_loader(self):
        with open("inputs/input.txt", "r") as file:
            return list(map(self.create_pair, file.read().splitlines()))

    def explode(self, pair, bal=0):
        if isinstance(pair,int):
            if bal>=4:
                return 0, True
            return pair, False
        pair.left, bleft = self.explode(pair.left, bal+1)
        pair.right, bright = self.explode(pair.right, bal+1)
        return pair, bleft or bright
    
    def add(self, snail1, snail2):
        return Pair(snail1, snail2)
    
    def split(self, pair):
        if isinstance(pair, int):
            if pair>9:
                return Pair(pair//2, (pair+1)//2), True
            return pair, False
        pair.left, bleft = self.split(pair.left)
        pair.right, bright = self.split(pair.right)
        return pair, bleft or bright

    def create_pair(self, snail):
        stack = [] # Stack that holds my Pair objects. 
        prevPair = None
        for i in range(len(snail)):
            if snail[i]=='[':
                stack.append(Pair(-1, -1)) # -1 because None is equivalent to 0, so it fails a check
            elif snail[i]==']':
                prevPair = stack.pop()
                if stack and stack[-1].left==-1:
                    stack[-1].left = prevPair
                elif stack and stack[-1].right==-1:
                    stack[-1].right = prevPair
            elif snail[i].isdigit():
                if stack[-1].left==-1:
                    stack[-1].left = int(snail[i])
                else:
                    stack[-1].right = int(snail[i])
        return prevPair

    def snailnumbers_sum(self, left_pair, right_pair):
        if isinstance(left_pair, int) and isinstance(right_pair, int):
            return 3*left_pair + 2*right_pair
        if isinstance(left_pair, int):
            return 3*left_pair + 2*self.snailnumbers_sum(right_pair.left, right_pair.right)
        if isinstance(right_pair, int):
            return 3*self.snailnumbers_sum(left_pair.left, left_pair.right) + 2*right_pair
        return 3*self.snailnumbers_sum(left_pair.left, left_pair.right) + 2*self.snailnumbers_sum(right_pair.left, right_pair.right)

    def run(self):
        initial_pair = self.numbers[0]
        print(initial_pair)
        print(self.explode(initial_pair))
        for i in range(1, len(self.numbers)):
            initial_pair = self.add(initial_pair, self.numbers[i])
            while True:
                isReduce = False
                initial_pair, isReduce = self.explode(initial_pair)
                if isReduce:
                    continue
                initial_pair, isReduce = self.split(initial_pair)
                if not isReduce:
                    break
        return self.snailnumbers_sum(initial_pair.left, initial_pair.right)

if __name__ == '__main__':
    s = SnailNumber()
    print(s.run())
    # s.explode(explode_test)

"""
My own test case for the exploding 
'[[[[3,3]]]]'
'[[[[[3,3]]]]]'

expecting [[[[8,7],[7,7]],[[8,6],[7,7]]],[[[0,7],[6,6]],[8,7]]]
"""