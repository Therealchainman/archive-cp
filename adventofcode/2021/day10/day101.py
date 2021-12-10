class SyntaxScoring:
    def __init__(self):
        self.errorScore = 0
        self.openings = {'{': '}', '(': ')', '[': ']', '<': '>'}
        self.points = {k: v for k, v in zip((')', ']', '}', '>'), (3, 57, 1197, 25137))}
    def isCorrupted(self, chunk):
        stk = []
        for c in chunk:
            if c in self.openings:
                stk.append(c)
            else:
                prev = stk.pop()
                if self.openings[prev] != c:
                    self.errorScore += self.points[c]
                    return True
        return False
    def dataLoader(self):
        with open("inputs/input.txt", "r") as f:
            return f.read().splitlines()
    def run(self):
        data = self.dataLoader()
        for chunk in data:
            self.isCorrupted(chunk)
        return self.errorScore
if __name__ == '__main__':
    print(SyntaxScoring().run())