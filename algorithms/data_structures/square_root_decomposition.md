# Square Root Decomposition

Square root decomposition is the simplest “preprocess to answer fast” pattern

With sqrt decomposition for RMQ, you split the array into blocks of size ~√n, store each block’s minimum, then answer a query by:
- scanning the partial block on the left,
- scanning the partial block on the right,
- taking mins of full blocks in the middle.

That immediately makes the core concept concrete:
- precompute summaries
- pay a bit up front
- queries get faster
- you can tune block size to balance build vs query time

Sparse Table is also preprocessing, but it’s less intuitive at first because the preprocessing is “all powers of two” and the query trick is “two overlapping intervals”.


## Square Root with small k/large k trick

You use two different algorithms for the queries with small and large k. 

For example, you use a naive approach for large k, while for small k, you use a more sophisticated method involving residue-class difference arrays. (just for example)

