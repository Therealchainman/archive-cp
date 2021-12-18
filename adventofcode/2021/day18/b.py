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
            return file.read().splitlines()

    def explode(self, snail):
        bal, i = 0, 0
        left = None
        val = 0
        while i<len(snail):
            if snail[i]=='[':
                bal += 1
                i+=1
            elif snail[i]==']':
                bal -= 1
                i+=1
            elif snail[i]==',':
                i+=1
            elif bal==5:
                index = i
                x, y = 0, 0
                while snail[i].isdigit():
                    x = x*10 + int(snail[i])
                    i += 1
                i += 1
                while snail[i].isdigit():
                    y = y*10 + int(snail[i])
                    i += 1
                i += 1
                index2 = i
                right = None
                rv = 0
                while i<len(snail):
                    start = i
                    while snail[i].isdigit():
                        rv = rv*10 + int(snail[i])
                        i += 1
                    end = i-1
                    if start<=end:
                        right = (start, end)
                        break
                    i += 1
                # print(left,right)
                if left and right:
                    res = snail[:left[0]] + str(val+x) + snail[left[1]+1:index-1] + '0' + snail[index2:right[0]] + str(rv+y) + snail[right[1]+1:]
                    # print(res)
                    return res, True
                if left:
                    res = snail[:left[0]] + str(val+x) + snail[left[1]+1:index-1] + '0' + snail[index2:]
                    # print(res)
                    return res, True
                if right:
                    res = snail[:index-1] + '0' + snail[index2:right[0]] + str(rv+y) + snail[right[1]+1:]
                    # print(res)
                    return res, True
                res = snail[:index-1] + '0' + snail[index2:]
                # print(res)
                return res, True
            else:
                start = i
                val = 0
                while i<len(snail) and snail[i].isdigit():
                    val = val*10 + int(snail[i])
                    i+=1
                left = (start, i-1)
        return snail, False
                
    
    def add(self, snail1, snail2):
        return '[' + snail1 + ',' + snail2 + ']'
    
    def split(self, snail):
        i = 0
        while i<len(snail):
            start = i
            val = 0
            while snail[i].isdigit():
                val = val*10 + int(snail[i])
                i+=1
            end  = i-1
            if val>9:
                res = snail[:start] + '[' + str(val//2) + ',' + str((val+1)//2) + ']' + snail[end+1:]
                # print(res)
                return res, True
            i+=1
        return snail, False

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
        n = len(self.numbers)
        max_magnitude = 0
        for i in range(n):
            for j in range(n):
                if i==j:
                    continue
                cur_sum = self.add(self.numbers[i], self.numbers[j])
                while True:
                    isReduce = False
                    cur_sum, isReduce = self.explode(cur_sum)
                    if not isReduce:
                        cur_sum, isReduce = self.split(cur_sum)
                    if not isReduce:
                        break
                initial_pair = self.create_pair(cur_sum) # creates pair object from the string
                magnitude = self.snailnumbers_sum(initial_pair.left, initial_pair.right)
                max_magnitude = max(max_magnitude, magnitude)
        return max_magnitude
if __name__ == '__main__':
    s = SnailNumber()
    print(s.run())

"""
My own test case for the exploding 
'[[[[3,3]]]]'
'[[[[[3,3]]]]]'

expecting [[[[8,7],[7,7]],[[8,6],[7,7]]],[[[0,7],[6,6]],[8,7]]]
"""