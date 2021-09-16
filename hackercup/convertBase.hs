{- I think this is how to comment in haskell, here is an example of converting a number from 
a base to another base, convert a number n from base b1 to base b2 with haskell. 
This is a reference for learning haskell -}
import Data.List (elemIndex)
import Data.Maybe (fromJust)
import Numeric (showIntAtBase)

alphabet = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
val c = fromJust . elemIndex c $ alphabet

main = do
    n <- getLine
    b1 <- readLn :: IO Int
    b2 <- readLn :: IO Int
    
    putStrLn $ (showIntAtBase b2 (alphabet!!) $ readBase b1 n)""

readBase b n = go 0 b n 
 where
  go a _ [] = a
  go a b (x:xs) = go (a*b+val x) b xs