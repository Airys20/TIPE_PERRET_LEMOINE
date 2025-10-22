import json_utils as ju
import numpy as np


catalogue = "catalogue.json"

def distance(i,M): # avec i : l'emplacement de la minutie de réf dans M ; M : l'ensemble des coordonnées de l'empreinte  
    (x0,y0)   = M[i]['coordonnees']
    distances = [] 
    n = len(M)
    if (i>100):
        j0 = i-100
    else : 
        j0 = 0
    j = j0
    while (j<(min((100+i), (n-1)))) :
        (x,y) = M[j]['coordonnees']
        if(x==x0 and y==y0):
            distances.append(0)
        else:
            distances.append(np.sqrt((x0-x)**2 + (y0-y)**2))
        j = j+1    
    temp = sorted(distances)
    dist_sort = []
    for k in range(len(temp)):
        dist_sort.append(j0 + distances.index(temp[k]))
    
    return(dist_sort)



'''
TEST :
M0 = (ju.get_data(0, catalogue))['minutiae']
print(distance(0, M0) ) 

'''


def neighborhoods(M): 
    nbh = []
    n = len(M)
    for i in range(n):
        dist  = distance(i,M) 
        nbh_i = []
        nbh_i.append((dist[0],dist[1], dist[2]))
        nbh_i.append((dist[0],dist[1], dist[3])) 
        nbh_i.append((dist[0],dist[2], dist[3]))            
        nbh.append(nbh_i)
    return(nbh)

'''
TEST : 
M0 = (ju.get_data(0, catalogue))['minutiae']
print(neighborhoods(M0))

'''

def kabsch_umeyama(A, B):  #article : Aligning point patterns with Kabsch–Umeyama algorithm by Tuomas Siipola
    assert A.shape == B.shape
    n, m = A.shape

    EA = np.mean(A, axis=0)
    EB = np.mean(B, axis=0)
    VarA = np.mean(np.linalg.norm(A - EA, axis=1) ** 2)

    H = ((A - EA).T @ (B - EB)) / n
    U, D, VT = np.linalg.svd(H)
    d = np.sign(np.linalg.det(U) * np.linalg.det(VT))
    S = np.diag([1] * (m - 1) + [d])

    R = U @ S @ VT
    c = VarA / np.trace(np.diag(D) @ S)
    t = EA - c * R @ EB

    return R, c, t
 
#prend une matrice en entrée : np.array([[][]])

def comparaison(Nc, Nr, Mc, Mr) :
    
    #creation de la matrice des voisins du catalogue 
    (x0,y0) = Mc[Nc[0]]
    (x1,y1) = Mc[Nc[1]]
    (x2,y2) = Mc[Nc[2]]
    
    Nc_matrix = np.array([[ x0, y0],
              [ x1, y1],
              [ x2, y2]])
    
    #creation de la matrice des voisins de recherche 
    (x0,y0) = Mr[Nr[0]]
    (x1,y1) = Mr[Nr[1]]
    (x2,y2) = Mr[Nr[2]]
    
    Nr_matrix = np.array([[ x0, y0],
              [ x1, y1],
              [ x2, y2]])
    
    #calcul des parametres 
    (R,c,t) = kabsch_umeyama(Nc_matrix, Nr_matrix)
    
    #local matching decision 
    #scaling comparaison 
    
    #sum of squared distances between the corresponding minutiae 
    #differences of orientation 
    
    return (#booléen , R, c, t )
        
def emm_alg(): 
    