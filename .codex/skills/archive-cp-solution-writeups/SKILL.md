---
name: archive-cp-solution-writeups
description: Use this skill when asked to explain a competitive programming solution, write the prose around contest code, fill in explanation sections for LeetCode/Codeforces/AtCoder markdown notes, or turn a C++/Python submission into a short why-it-works breakdown with algorithm intuition, invariants, and code-to-idea mapping.
---

# Archive-CP Solution Writeups

Use this skill when the user gives one or more contest solutions and wants the explanation text that should sit above the code.

## Find the output shape first

1. If updating an existing contest file, read nearby entries and match their heading depth, keyword list style, and paragraph length.
2. Preserve structures like `## Qx. ...`, `### Solution 1: ...`, explanation paragraphs, then the fenced code block.
3. If the user only gives raw code, default to writing just the explanation block that can be pasted above the snippet.

## What the explanation should do

- Name the main technique in the `### Solution 1: ...` line.
- Start with the core idea in one or two sentences.
- Explain the key invariant, state definition, or graph/tree interpretation.
- Say why the important step is correct, not just what the code does.
- Map the main arrays, helper functions, or traversal order back to the idea.
- Mention an impossibility check, greedy justification, or subtle edge case when it matters.
- Mention complexity only when it adds value.

## What to avoid

- Do not narrate syntax line by line.
- Do not restate obvious operations like `reverse`, `sort`, or `push_back` unless the choice itself matters.
- Do not write a full formal proof unless the user asks for one.
- Do not hide the hard step behind vague phrases like "DFS handles it" or "DP takes care of the rest".

## Writing workflow

1. Infer the algorithm family from the code and problem title if available.
2. Identify the one observation that kills the brute force.
3. Find the invariant, DP meaning, greedy exchange, or traversal property that makes the algorithm correct.
4. Explain why the chosen order of processing is necessary or natural.
5. Explain how the code stores that idea.
6. Cut any sentence that only paraphrases syntax.

## Good explanation patterns

### Tree and DFS solutions

Explain what a subtree contributes and why the DFS order matters. If the code fixes children before the parent, say what becomes irrevocable after leaving a subtree and why that makes postorder the natural order.

For flip or parity problems on trees, a strong default explanation is:

- after processing child `v`, every node below `v` is already settled
- if `v` still mismatches, the edge from its parent is the only remaining move that can fix it
- taking that move is necessary, and it only changes the two endpoints tracked by the code

### Sliding window solutions

State exactly what information the window stores. Then explain why moving the right pointer updates that information locally, and why greedily moving the left pointer cannot destroy a better answer.

### Dynamic programming solutions

Define the DP state in plain English. Then explain why each transition covers a valid way to build the answer and why the iteration order guarantees prerequisites are ready.

### Greedy solutions

Name the local choice and give the exchange or monotonicity reason that makes it safe.

### Graph shortest path solutions

Describe the implicit state graph and why a path cost in that graph matches the problem objective.

## Code-to-idea mapping

After the main explanation, add a short variable mapping when it helps:

- `adj[u]` stores the graph or tree structure
- `dp[...]` stores the best value for a state
- `freq` stores the current window counts
- helper functions like `dfs`, `query`, or `relax` correspond to one logical step in the algorithm

This section should clarify non-obvious state, not document every variable.

## Default template

````md
### Solution 1: technique keywords

Short statement of the main idea and the key observation.

If an invariant or state definition matters, explain it in one short paragraph or a few flat bullets.

Explain the main transition, traversal, or greedy choice and why it is correct.

Briefly connect the important variables/functions in the code to that idea.

```cpp
// code
```
````

## Style for this archive

- Keep it concise and implementation-first.
- Usually aim for two to four short paragraphs total.
- Use flat bullets only when they make state definitions clearer.
- Prefer phrases like "The key observation is...", "This works because...", and "After that..." over theorem-style prose.
- Put interesting edge cases inline instead of adding a long separate section.

## Final checks

- Could a reader understand why the algorithm works without tracing every line?
- Did the explanation focus on the hard step rather than the easy syntax?
- If the code uses a non-obvious order, did you explain why that order is needed?
- If updating an existing markdown file, does the tone match nearby solutions?
