# Understanding Cycles in Rotations

Think of **n chairs in a circle**, labeled 1, 2, 3, …, n.

Now pick a **jump size i**. Press a button. Everyone stands up and moves **i chairs clockwise**.  
That single button press is the permutation **πᵢ**.

---

## What is a “cycle”?

Pick one chair, say chair **r**. Keep pressing the button and write down the chairs you land on:

```
r → r + i → r + 2i → r + 3i (mod n)
```

When you get back to r, stop. The list you made is one **cycle**.

Do the same starting from a chair you have not yet listed to get the next cycle.  
Repeat until all chairs are in some cycle.

> **A cycle** is just the loop you get by “keep jumping i chairs at a time.”

---

## What do these cycles look like?

Two simple patterns cover everything.

### 1. When i and n share no common factor except 1
Example: n = 7, i = 3

Starting anywhere and jumping by 3 hits **every chair** before you return.  
That means there is **one big cycle** with all n chairs.

### 2. When i and n share a common factor d > 1
Example: n = 12, i = 8 → gcd(12, 8) = 4

Jumping by 8 skips through only 1 out of every 4 chairs.  
You never touch chairs from the other 3 “tracks.”

So the circle splits into **d separate loops**:

```
(1 9 5)
(2 10 6)
(3 11 7)
(4 12 8)
```

Each loop has the same size.

---

## The simple rule

- **Number of cycles** = gcd(i, n) → “how many tracks”  
- **Length of each cycle** = n / gcd(i, n) → “chairs per track”

Why does this make sense?  
You have n chairs split evenly into d tracks, so each track holds n/d chairs.

---

## Quick checks

| πᵢ | n | gcd(i, n) | Number of Cycles | Cycle Length | Example Cycles |
|----|---|------------|------------------|---------------|----------------|
| π₀ | any | n | n | 1 | (1)(2)(3)…(n) |
| π₁ | any | 1 | 1 | n | (1 2 3 … n) |
| π₂ | 8 | 2 | 2 | 4 | (1 3 5 7)(2 4 6 8) |

---

## The big picture

This is exactly what’s happening in the **necklace-coloring** or **rotation** problems:

- Each πᵢ is a rotation of the positions (a symmetry of the circle).
- Burnside’s lemma counts how many colorings stay the same under each rotation.
- A coloring stays the same if each cycle is a single color → that’s why we raise the number of colors to the number of cycles.

So the math rule `C(πᵢ) = gcd(i, n)` is just a neat number-theory way of saying:  
> “How many independent loops do you make if you keep jumping i steps around an n-seat circle?”
