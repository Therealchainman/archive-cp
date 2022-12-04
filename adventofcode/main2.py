class Assignment:
    def __init__(self, sections: str):
        elf1, elf2 = sections.split(',')
        self.s1, self.e1 = map(int, elf1.split('-'))
        self.s2, self.e2 = map(int, elf2.split('-'))
def main():
    with open('input.txt', 'r') as f:
        data = f.read().splitlines()
        res = sum([1 for work in map(Assignment, data) if min(work.e1, work.e2) - max(work.s1, work.s2) >= 0])
        return res
if __name__ == "__main__":
    print(main())