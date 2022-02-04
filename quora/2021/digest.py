def make_2d(rows, cols, val=False):
    return [[val for j in range(cols)] for i in range(rows)]


N, M = map(int, input().split())
created_by = [0] + [int(input()) for i in range(N)]
P, Q = map(int, input().split())

follows_user = make_2d(M + 1, M + 1)
for i in range(P):
    a, b = map(int, input().split())
    follows_user[a][b] = True

follows_story = make_2d(M + 1, N + 1)
for i in range(Q):
    a, b = map(int, input().split())
    follows_story[a][b] = True

user_to_user = make_2d(M + 1, M + 1, 0)
user_to_story = make_2d(M + 1, N + 1, 0)
for i in range(1, M + 1):
    for j in range(1, M + 1):
        if i == j:
            continue
        if follows_user[i][j]:
            user_to_user[i][j] = 3
            continue
        for k in range(1, N + 1):
            if created_by[k] == j and follows_story[i][k]:
                user_to_user[i][j] = 2
                break
            if follows_story[j][k] and follows_story[i][k]:
                user_to_user[i][j] = 1

for i in range(1, M + 1):
    for j in range(1, N + 1):
        if created_by[j] == i:
            user_to_story[i][j] = 2
        elif follows_story[i][j]:
            user_to_story[i][j] = 1


for i in range(1, M + 1):
    scores = []
    for j in range(1, N + 1):
        score = 0
        if created_by[j] == i or follows_story[i][j]:
            score = -1
        else:
            score = sum([user_to_user[i][k] * user_to_story[k][j] for k in range(1, M+1)])
        scores.append((-score, j))
    scores.sort()
    print(" ".join([str(j) for _, j in scores[:3]]))