from functools import reduce
class SyntaxScoring:
    def __init__(self):
        self.openings = {'{': '}', '(': ')', '[': ']', '<': '>'}
        self.points = {k: v for k, v in zip((')', ']', '}', '>'), range(1,5))}
        self.closers = []
    def isCorrupted(self, chunk):
        """
        Returns true if it is corrupted
        if it is not corrupted it will add the sequence of closing characters
        to a global variable self.closings to make the line complete. 
        """
        stk = []
        for c in chunk:
            if c in self.openings:
                stk.append(c)
            else:
                prev = stk.pop()
                if self.openings[prev] != c:
                    return True
        self.closers = [self.openings[c] for c in stk[::-1]]
        return False
    def dataLoader(self):
        with open("inputs/input.txt", "r") as f:
            return f.read().splitlines()

    def run(self):
        """
        Returns the middle value in the scores for the necessary sequence of closing characters for
        the incomplete lines.
        """
        data = self.dataLoader()
        scores = [reduce(lambda a, b: a*5 + b, (self.points[c] for c in self.closers), 0) for chunk in data if not self.isCorrupted(chunk)]
        scores.sort()
        return scores[len(scores)//2]
if __name__ == '__main__':
    print(SyntaxScoring().run())