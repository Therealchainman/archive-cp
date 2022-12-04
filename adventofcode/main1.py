

def main():
    with open('input.txt', 'r') as f:
        data = f.read().splitlines()
        res = 0
        for elf1, elf2 in map(lambda x: x.split(','), data):
            s1, e1 = map(int, elf1.split('-'))
            s2, e2 = map(int, elf2.split('-'))
            res += (s1 >= s2 and e1 <= e2) or (s2 >= s1 and e2 <= e1)
        return res
if __name__ == "__main__":
    print(main())