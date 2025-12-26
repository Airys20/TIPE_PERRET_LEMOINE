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

def equation(U, V):
    S = s(U)
    l = U.shape[0]
    P = np.hstack([np.ones((l,1)), U])
    
    top = np.hstack([S, P])
    bottom = np.hstack([P.T, np.zeros((3,3))])
    L = np.vstack([top,bottom])    
    
    Y = np.vstack([V, np.zeros((3,2))])
    
    params = np.linalg.solve(L, Y)
    W = params[:1, :]
    c = params[1, :]
    A = params[l+1:, :]
    return W, c, A 
    
V = np.array([(5,6), (7,2), (3,4)])
print(equation(U,V))

data_empreinte = 'recherche.json'

def TPS(U, V ):
    with open(data_empreinte, "r", encoding="utf-8") as f :
        base = json.load(f)
    
    
    recherche = (base)
    
    nr = len(recherche)
    (W, c, A) = equation(U, V)
    
    
    def f(x,y) : 
        return  c + A * (x,y) + W.T * s(U)
    
    iter_coordonnees (recherche, f)
    
    return 
    
    
