# BITSET

If you have an array containing just 0 and 1s you can use a bitset to optimize it, and if length of array is n elements and it contained 64 bit integers, you can optimize it to n / 64 elements.

A bitset is an array of bools but each boolean value is not stored in a separate byte instead, bitset optimizes the space such that each boolean value takes 1-bit space only, so space taken by bitset is less than that of an array of bool or vector of bool. 

A limitation of the bitset is that size must be known at compile time i.e. size of the bitset is fixed.

reset() : This function is used to reset all bits of the bitset to 0.
reset(i): This function is used to reset the ith bit of the bitset.
set(i): This function is used to set the ith bit of the bitset.
