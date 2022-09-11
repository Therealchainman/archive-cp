import sys
problem = sys.argv[0].split('.')[0]
validation = ''
coord_thres = 3001
mod = int(1e9)+7

def find_distance_1_dim(trees, wells):
    result = 0
    for i in range(coord_thres):
        cnt_trees = trees[i]
        if cnt_trees == 0: continue
        for j in range(coord_thres):
            cnt_wells = wells[j]
            coef = (cnt_trees*cnt_wells)%mod
            term = (coef*(i-j)*(i-j))%mod
            result = (result + term)%mod
    return result
            
def main():
    N = int(f.readline())
    trees_x, trees_y = [0]*coord_thres, [0]*coord_thres
    wells_x, wells_y = [0]*coord_thres, [0]*coord_thres
    for _ in range(N):
        a, b = map(int, f.readline().split())
        trees_x[a] += 1
        trees_y[b] += 1
    Q = int(f.readline())
    for _ in range(Q):
        x, y = map(int, f.readline().split())
        wells_x[x] += 1
        wells_y[y] += 1
    result = find_distance_1_dim(trees_x, wells_x)%mod
    result = (result + find_distance_1_dim(trees_y, wells_y))%mod
    return result

if __name__ == '__main__':
    result = []
    with open(f'inputs/{problem}_{validation}input.txt', 'r') as f:
        T = int(f.readline())
        for t in range(1,T+1):
            result.append(f'Case #{t}: {main()}')
    with open(f'outputs/{problem}_output.txt', 'w') as f:
        f.write('\n'.join(result))