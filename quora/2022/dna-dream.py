"""
DNA dream brute force solution
"""

class DNA:
    def data_loader(self):
        _ = input()
        self.dna = input()
        # with open("inputs/input.txt", "r") as f:
        #     _ = f.readline()
        #     self.dna = f.readline().strip()

    def run(self):
        self.data_loader()
        n = len(self.dna)
        bonds_dict = {'A':'T', 'T':'A', 'G':'C','C':'G'}
        maxBonds = 0
        basesLeft = 0
        for i in range(n):
            k = 2*i+1
            bonds = 0
            for j in range(i+1):
                if k<n and bonds_dict[self.dna[j]]==self.dna[k]:
                    bonds+=1
                k-=1
            if bonds>maxBonds:
                maxBonds=bonds
                basesLeft=i+1
        print(basesLeft, maxBonds)
if __name__ == '__main__':
    DNA().run()