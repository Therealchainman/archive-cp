

# Part 1 and 2

I store offline all the possible transitions from pair of characters to the two pair of characters it creates.  
I store this in a dictionary called transitions.  Then I create a Counter called freqs for storying the count of each pair of characters from template.  

I create a third Counter to store the counts of each character.  

Then all we have to do is iterate through the 100 possible transitions to create the next freqs and increase the count for the inserted character.  

```py
from collections import Counter
with open("inputs/input.txt", "r") as f:
    freqs = Counter()
    template = f.readline().strip()
    f.readline()
    raw_data = f.read().splitlines()
    for k, v in zip(template, template[1:]):
        s = k + v
        freqs[s]+=1
    transitions = {}
    for line in raw_data:
        x, y = line.split(" -> ")
        transitions[x] = [x[0]+y, y+x[1]]
    counts = Counter(template)
    for _ in range(40):
        tmp = Counter()
        for pair, count in freqs.items():
            if pair in transitions:
                counts[transitions[pair][0][1]] += count
                for new_pair in transitions[pair]:
                    tmp[new_pair] += count
        freqs = tmp
    mostCommon = counts.most_common()
    print(mostCommon[0][1]-mostCommon[-1][1])
```