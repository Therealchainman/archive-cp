# Unranking

An unranking algorithm is a computational technique that takes an integer rank (position) as input and returns the corresponding combinatorial object (e.g., permutation, combination, tree) from an ordered set.

## A general strategy

Turn the question “which object is at position r?” into repeated counting + skipping of whole blocks.

1. Choose ordering and a "Prefix"

You need a total order (lexicographic, numeric, colex, Gray code, etc.). Then describe objects by building them step by step (a prefix that you extend).

Examples of a prefix:
- Permutations: first element, then second, ...
- combinations: decide include/exclude for each candidate elemtn
- binary strings with K ones: decide bits for MSB to LSB

2. Know how to count completions of a prefix

This is key and needs to be a fast function

count(prefix) = number of valid full objects that start with this prefix.

3. Walk the decision tree using "rank vs block size"

At each step you consider options in order (the same order your global ordering implies).

For each option:

compute how many objects it wold generate (a block size)
- If rank > block_size: subtract and skip that entire block
- else: comit to that option and continue with the same rank.

Then compare the rank against the block size.

For 1-based rank:

if r > blockSize:
    r -= blockSize
    skip this choice
else:
    choose this option
    continue building

For 0-based rank:

if r >= blockSize:
    r -= blockSize
    skip this choice
else:
    choose this option
    continue building

This is the heart of unranking.

You are walking the decision tree, but you skip whole subtrees whenever the rank is outside them.

4. Keep the rank convention consistent

Decide whether rank is 0-based or 1-based and stick tpo it. 

- 0-based: skip while rank >= block
- 1-based: skip while rank > block

## Final tip

At each decision point:
    count how many objects each choice represents
    skip choices whose entire block comes before the target rank
    choose the first block containing the rank