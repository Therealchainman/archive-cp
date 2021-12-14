# sys.stdout = open('outputs/output.txt', 'w')
class SevenSegmentSearch:
    def dataLoader(self, path = "inputs/input.txt"):
        with open("inputs/input.txt", "r") as f:
            loadedData = f.read().splitlines()
        for data in loadedData:
            rawPatterns, rawDigits = map(str.split, data.split('|'))
            patterns = tuple(map(lambda x: (frozenset(x), len(x)), rawPatterns))
            digits = tuple(map(lambda x: frozenset(x), rawDigits))
            yield patterns, digits

    def deduceMapping(self, patterns):
        # pattern to digit mapping
        p2d = dict()
        for p, plen in patterns:
            if plen == 2:
                p2d[p] = 1
            elif plen == 3:
                p2d[p] = 7
            elif plen == 4:
                p2d[p] = 4
            elif plen == 7:
                p2d[p] = 8
        d2p = {v: k for k, v in p2d.items()}
        for p, plen in patterns:
            # 3 or 5 or 2, all have length of 5
            if plen==5:
                if len(p & d2p[1]) == 2:
                    p2d[p] = 3
                elif len(p&d2p[4]) == 3:
                    p2d[p] = 5
                else:
                    p2d[p] = 2
            elif plen==6:
                if len(p&d2p[4])==4:
                    p2d[p] = 9
                elif len(p&d2p[7])==2:
                    p2d[p] = 6
                else:
                    p2d[p] = 0
        return p2d

    def getOutput(self):
        totalCnt = 0
        for pattern, digit in self.dataLoader():
            p2d = self.deduceMapping(pattern)
            cnt = 0
            for dig in digit:
                cnt = cnt*10 + p2d[dig]
            totalCnt += cnt
        return totalCnt
# sys.stdout.close()

if __name__ == "__main__":
    s = SevenSegmentSearch()
    print(s.getOutput())