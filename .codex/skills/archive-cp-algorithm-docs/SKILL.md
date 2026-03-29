---
name: archive-cp-algorithm-docs
description: Use this skill when asked to create or update competitive programming algorithm documentation that should match the user's existing archive-cp algorithms folder. It chooses the right category, mirrors the current markdown structure, and writes concise implementation-first notes with practical code snippets.
---

# Archive-CP Algorithm Docs

Use this skill when the user wants a new algorithm note, wants an existing algorithm note expanded, or asks for documentation that should match the structure already used in the `algorithms` folder.

## Find the target folder first

1. Prefer a repo-local `algorithms/` directory if it already contains the competitive-programming note categories.
2. Otherwise, look for the user's `archive-cp/algorithms/` repo nearby and write there.
3. Reuse the existing top-level taxonomy instead of inventing a new one.

Current category set:

- `bitwise_and_subsets`
- `combinatorics`
- `data_structures`
- `dynamic_programming`
- `game_theory`
- `geometry`
- `graph_theory`
- `mathematics`
- `misc`
- `number_theory`
- `probability`
- `scripts`
- `strings`

Do not create a new top-level category unless the user explicitly asks for it.

## Match the existing file pattern

- The default output is a single markdown file in the chosen category.
- Use a lowercase snake_case filename ending in `.md`.
- Before creating a new file, check whether the topic already exists under a nearby or slightly broader name. Extend the existing note instead of creating a duplicate when overlap is substantial.
- Keep titles simple. Start with one `#` heading for the algorithm name. Usually this is title case, but if nearby notes use lowercase or acronym-heavy naming, match that local style.

## Mirror the writing style

- Write like a competitive-programming notebook, not a textbook chapter.
- Keep explanations short, practical, and implementation-first.
- Lead with the idea that helps solve problems, then show the code quickly.
- Use informal but clear prose. A little shorthand is fine if the idea stays understandable.
- Skip long historical background, citations, and full formal proofs unless the user asks for them.
- Mention complexity, indexing assumptions, invariants, or edge cases when they matter for using the technique correctly.

## Use the same section structure

Most existing notes follow this rhythm:

1. `#` Title
2. One or more `##` sections for the main algorithm, a useful variant, or a common application
3. Optional short explanation paragraphs under each section
4. A fenced code block soon after the explanation
5. Optional extra sections for alternate implementations, derivations, or worked usage

Use `###` only when a subsection truly helps. Most notes do not need deep nesting.

## Code expectations

- Prefer `cpp` snippets unless Python is clearly the better fit or the neighboring docs for that topic mostly use Python.
- Snippets should be ready to lift into a contest solution.
- Include just enough surrounding code to make the technique understandable. Full `main()` boilerplate is optional.
- If there are multiple useful variants, separate them with their own `##` sections or short lead-ins.
- Call out assumptions such as `0`-indexed arrays, `1`-indexed nodes, required preprocessing, or expected constraints when those assumptions affect correctness.

## Working template

Use this as the default shape, then adjust to match neighboring notes:

````md
# Topic Name

## Core algorithm or main trick

Short practical explanation of what the technique computes, when to use it, and the one or two key invariants.

```cpp
// implementation
```

## Common variant or application

Short explanation of what changes and why.

```cpp
// implementation
```
````

## Before you finalize

- Read two or three nearby notes in the same category and align to their capitalization, terseness, and code density.
- Make sure the category and filename are consistent with the rest of the archive.
- Prefer expanding an existing note over splitting one topic across many tiny files.
- Keep the final result useful as a quick-reference note for future problem solving.

## When updating an existing note

- Preserve the current voice and structure unless the user asked for a cleanup.
- Add new variants in separate `##` sections instead of rewriting everything.
- Do not remove working code examples just to make the document more polished.
- If a note is clearly inconsistent or confusing, improve it gently while keeping the archive's practical style.
