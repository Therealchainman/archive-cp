

I don't know I just brute forced the solution with some arbitrary bounds.  Here is
a better solution I learned online that I used as my final cause it is much better.  

The good use of regex and everything.  Also uses a math for part 1 answer to find the maximum y height.   


```py
import re
with open("inputs/input.txt", "r") as f:
    xmin, xmax, ymin, ymax = list(map(int, re.findall(r'[-\d]+', f.read())))
    cnt = 0
    print(abs(ymin)*abs(ymin+1)//2) # part 1
    for dx_init in range(min(0, xmin-1), max(0, xmax) + 1):
        for dy_init in range(ymin,abs(ymin)+1):
            dx,dy=dx_init,dy_init
            x, y = 0, 0
            while y>ymin:
                y+=dy
                x+=dx
                if dx>0:
                    dx-=1
                if dx<0:
                    dx+=1
                dy-=1
                if xmin<=x<=xmax and ymin<=y<=ymax:
                    cnt+=1
                    break
    print(cnt) # part 2
        
```