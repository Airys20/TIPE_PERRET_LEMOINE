import math
import numpy as np

def sigma(u) :
    (x,y) = u
    if(x==0 and y==0) :
        return 0
    else:  
        return (x**2 + y**2)*math.log10(np.sqrt((x**2) + (y**2))) 


def s(U) :
    l = len(U)
    res = np.zeros((l,l)) 
    for j in range(l) :
        for k in range(l) :
            (xj, yj) = U[j]
            (xk, yk) = U[k]
            res[j][k] = sigma(((xj - xk) , (yj - yk)))
    return res 

U = np.array([(1,2), (2,3), (5,6)])
print(s(U))

