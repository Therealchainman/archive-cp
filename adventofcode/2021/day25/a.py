class SeaCucumber:
    def data_loader(self):
        with open("inputs/input.txt", "r") as f:
            return [list(line) for line in f.read().split('\n')]
    def run(self):
        data = self.data_loader()
        steps, R, C = 0, len(data), len(data[0])
        while True:
            ndata = [line[:] for line in data]
            for i in range(R):
                for j in range(C):
                    if ndata[i][j]=='>' and ndata[i][(j+1)%C]=='.':
                        ndata[i][j]='.'
                        ndata[i][(j+1)%C]='x'
            for j in range(C):
                for i in range(R):
                    if ndata[i][j]=='v' and ndata[(i+1)%R][j]=='.':
                        ndata[i][j]='.'
                        ndata[(i+1)%R][j]='y'
            for i in range(R):
                for j in range(C):
                    if ndata[i][j]=='x':
                        ndata[i][j]='>'
                    if ndata[i][j]=='y':
                        ndata[i][j]='v'
            if all(x==y for x,y in zip(data,ndata)):
                break
            steps += 1
            data = ndata
        print(steps-1)
if __name__ == '__main__':
    SeaCucumber().run()