from collections import Counter
with open("inputs/input.txt", "r") as f:
    freqs = Counter()
    template , raw_data = f.read().split('\n\n')
    print(template,raw_data)
    for k, v in zip(template, template[1:]):
        s = k + v
        freqs[s]+=1
    transitions = {}
    raw_data = raw_data.split('\n')
    for line in raw_data:
        x, y = line.split(" -> ")
        transitions[x] = [x[0]+y, y+x[1]]
    counts = Counter(template)
    for _ in range(10):
        tmp = Counter()
        for pair, count in freqs.items():
            if pair in transitions:
                counts[transitions[pair][0][1]] += count
                for new_pair in transitions[pair]:
                    tmp[new_pair] += count
        freqs = tmp
    mostCommon = counts.most_common()
    print(mostCommon[0][1]-mostCommon[-1][1])