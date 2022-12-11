import operator
import functools

class Monkey:
    def __init__(self, items, ops, div, if_true, if_false):
        self.items = items
        self.ops = ops
        self.div = div
        self.if_true = if_true
        self.if_false = if_false

    # determines what monkey to throw the item to
    def throw(self, val: int) -> int:
        return self.if_true if val%self.div == 0 else self.if_false

    def apply_ops(self, val: int) -> int:
        op, operand = self.ops
        if op == '+':
            return val + int(operand)
        elif operand == 'old':
            return val*val
        return val*int(operand)

def main():
    with open('input.txt', 'r') as f:
        data = f.read().splitlines()
        monkeys = []
        divisors = set()
        for i in range(7, len(data)+8, 7):
            monk = data[i-7:i]
            items = list(map(int, monk[1].replace(',', '').split()[2:]))
            ops = monk[2].split()[-2:]
            div = int(monk[3].strip().split()[-1])
            divisors.add(div)
            if_true = int(monk[4].strip().split()[-1])
            if_false = int(monk[5].strip().split()[-1])
            monkeys.append(Monkey(items, ops, div, if_true, if_false))
        inspect = [0]*8
        num_rounds = 10000
        for _ in range(num_rounds):
            for i, m in enumerate(monkeys):
                while m.items:
                    val = m.items.pop()
                    val = m.apply_ops(val)//3
                    monkeys[m.throw(val)].items.append(val)
                    inspect[i] += 1
        inspect.sort(reverse = True)
        return functools.reduce(operator.mul, inspect[:2])
if __name__ == "__main__":
    print(main())