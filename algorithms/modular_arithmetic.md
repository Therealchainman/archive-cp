# Modular Arithmetic


## Modular Congruence

modular congruence means that two integers are in the same equivalence class under some (mod m).  Imagine = is congruence symbol.  a = b(modm) implies that with division algorithm a = mq + r, where 0 <= r < m.  And b = mq + r, and both have the same remainder.  So when divided by m they have same remainder.  

It is somewhat obvious by definition of modular congruence that if you do this

a = b(modm) => a + m = b(modm) => a = (b+m)(modm)

If you add m either side they will still be in the same equivalence class cause they are just wrapping around at m.  So it still has same remainder.

Another property is 

 a*b = (a(modm) * b(modm))(modm) 

That is the multiplication of two integer a and b will be in the same congruence class as the multiplication of a and b modulo m.  When they are all under the modulus m. 