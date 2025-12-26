import numpy as np
import json_utils as ju 



''' *************************************** NEIGHBORS *************************************** '''

def chgt_base(M,t):
    xm, ym, theta_m = M[t][0:2]
    M_new = []
    for i in range(len(M)):
        xi, yi = M[i][0:1]
        xi = xi - xm 
        yi = yi - ym 
        xtmp = xi 
                
        xi = np.cos(theta_m)*xi + np.sin(theta_m)
        yi = - np.sin(theta_m)*xtmp + np.cos(theta_m)
        
        M_new.append(xi,yi)        
    return(M_new)
 
 
def fusion(L1,L2):
    n1 = len(L1)
    n2 = len(L2)
    L12 = [0]*(n1+n2)
    i1 = 0
    i2 = 0
    i = 0
    while i1<n1 and i2<n2:
        if L1[i1][0] < L2[i2][0]:
            L12[i] = L1[i1]
            i1 += 1
        else:
            L12[i] = L2[i2]
            i2 += 1
        i += 1
    while i1<n1:
        L12[i] = L1[i1]
        i1 += 1
        i += 1
    while i2<n2:
        L12[i] = L2[i2]
        i2 += 1
        i += 1 
    return L12

def tri_fusion_recursif(L):
    n = len(L)
    if n > 1:
        p = int(n/2)
        L1 = L[0:p]
        L2 = L[p:n]
        tri_fusion_recursif(L1)
        tri_fusion_recursif(L2)
        L[:] = fusion(L1,L2)

 
    
def distances(M): #on projette les minuties sur l'axe vertical et on les tries par distance. 
    
    #création d'un nouveau tableau M_new[i] = ( yi, i)
    M_new = []
    for i in range(len(M)):
        M_new.append = (M[i][1], i)
    
    #tri fusion sur yi 
    tri_fusion_recursif(M_new)
    return M_new 

def neighbors(M , nb_nghbr, i):   
    
    M_tmp = chgt_base(M, i)
    M_new = distances(M_tmp)
        
    n0 = []
    for j in range(nb_nghbr):
        n0.append(M_new[1+j][1])
    
    return (n0)
    
    
''' *************************************** SUBSTRUCTURES *************************************** '''
lim_theta = 0.5
nb_nghbr = 5

def corresponding_substruct(substruct_c, substruct_r): #PAS ENCORE TESTEE 
    bool = True
    for i in range(len(substruct_c)):
        while j < (len(substruct_c)):
            xi, yi = substruct_c[0:1] 
            xj, yj = substruct_r[0:1]
            
            m = (yi-yj)/(xi-xj)
            p = yi - m*xi 
            acc = 0
            
            for t in range(len(substruct_c)):
                if is_above(m,p,substruct_c[t][0], substruct_c[t][0]):
                    if is_above(m,p,substruct_r[t][0], substruct_c[t][0]):
                       acc = acc+1
                else :
                    if (is_above(m,p,substruct_r[t][0], substruct_c[t][0]) == False):
                        acc = acc+1
            if (acc != len(substruct_c)) : 
                 bool = False; 
            j = j+1
    return bool


def substructures(M_recherche, M_catalogue):
    b = False
    for i in range(len(M_recherche)):
        for j in range(len(M_catalogue)):      
            ar, br = M_recherche[i][2]
            ac, bc = M_catalogue[j][2]
            
            theta_r = np.arctan(br/ar)
            theta_c = np.arctan(bc/ac)
            
            if(abs(theta_r - theta_c) < lim_theta):
                #the two minutiaes are regarded to be the corresponding minutiaes 
                substruct_c = neighbors(M_catalogue, nb_nghbr, j)
                substruct_r = neighbors(M_recherche, nb_nghbr, i)
                
                if corresponding_substruct(substruct_c, substruct_r):
                    b = True
                    break 
            
    return(b, substruct_c, substruct_r)
    


#def sign(p1, p2, m): 
 
def is_above(m,p,xt,yt ):
    if ((m*xt + p)<yt) :
        return(True)
    else :
        return(False)
    return()

    

                

''' *************************************** ALIGNEMENT *************************************** '''

'''$$
m^* = T \times m
$$
$$ x_i^* = \sum_{k = 1}^{d} \sum_{l = 1}^{d} a_{kl}x_i^k y_i^l $$  $$ y_i^* = \sum_{k = 1}^{d} \sum_{l = 1}^{d} b_{kl}x_i^k y_i^l $$
'''


def polynomial_terms_2d(x, y, d):
    terms = []
    for p in range(d+1):
        for q in range(d+1-p):
            terms.append((p, q))
    vals = [ (x**p) * (y**q) for (p,q) in terms ]
    return vals

def estimate_polynomial_transform(src_pts, dst_pts, d):
    src_pts = np.asarray(src_pts)
    dst_pts = np.asarray(dst_pts)
    N = src_pts.shape[0]
    # nombre de monomes
    M = (d+1)*(d+2)//2
    if N < M:
        raise ValueError(f"Pas assez de points pour estimer un polynôme d'ordre {d}. Nécessite >= {M} pts, obtenu {N}.")

    # Construire matrice A (N x M) design pour monomes; on fera concat pour x et y indépendants
    A = np.zeros((N, M))
    for i in range(N):
        x, y = src_pts[i]
        A[i, :] = polynomial_terms_2d(x, y, d)

    bx = dst_pts[:, 0]
    by = dst_pts[:, 1]

    # solution least squares
    coeffs_x, *_ = np.linalg.lstsq(A, bx, rcond=None)
    coeffs_y, *_ = np.linalg.lstsq(A, by, rcond=None)

    # Construire la liste de (p,q) pour interprétation
    terms = []
    for p in range(d+1):
        for q in range(d+1-p):
            terms.append((p, q))

    return coeffs_x, coeffs_y, terms

def apply_polynomial_transform(minutiae, coeffs_x, coeffs_y, terms):
    for m in minutiae:
        x, y = m["coordonnes"]
        mon = [ (x**p)*(y**q) for (p,q) in terms ]
        x_new = float(np.dot(coeffs_x, mon))
        y_new = float(np.dot(coeffs_y, mon))
        m["coordonnees"] = [x_new, y_new]
    return()

def main_align(data_catalogue, data_recherche, i):
    base_catalogue = ju._charger_base(data_catalogue)
    base_recherche = ju._charger_base(data_recherche)
    
    
    dico_catalogue = base_catalogue[i]
    M_catalogue = []
    for j in range(len(dico_catalogue)):
        xi = dico_catalogue["minutiae"][j]["coordonnees"][0]
        yi = dico_catalogue["minutiae"][j]["coordonnees"][1]
        theta_i = dico_catalogue["minutiae"][j]["orientation"]
        M_catalogue.append((xi, yi, theta_i))
        
    dico_recherche = base_recherche[0]
    M_recherche = []
    for j in range(len(dico_recherche)):
        xi = dico_recherche["minutiae"][j]["coordonnees"][0]
        yi = dico_recherche["minutiae"][j]["coordonnees"][1]
        theta_i = dico_recherche["minutiae"][j]["orientation"]
        M_recherche.append((xi, yi, theta_i))
    
    b, substruct_c, substruct_r = substructures(M_recherche, M_catalogue)
    
    coeffs_x, coeffs_y, terms = estimate_polynomial_transform(substruct_c, substruct_r , nb_nghbr)
    apply_polynomial_transform(dico_recherche["minutiae"], coeffs_x, coeffs_y, terms)
    
    return()    
                 
''' *************************************** TEST *************************************** '''         
main_align('catalogue.json', 'recherche.json', 2 )
                
''' *************************************** BROUILLON *************************************** '''
'''
def neighbors(data_set , nb_nghbr):

    base = ju._charger_base(data_set)
    
    for i in range(len(base)):
        dico_personne = base[i]
        M = []
        for j in range(len(dico_personne)):
            xi = dico_personne["minutiae"][j]["coordonnees"][0]
            yi = dico_personne["minutiae"][j]["coordonnees"][1]
            theta_i = dico_personne["minutiae"][j]["orientation"]
            n0 = dico_personne["minutiae"][j]["neighbors"]
        
            M.append((xi, yi, theta_i, n0))
        M = chgt_base(M, i)
        M_new = distances(M)
        
        n0 = []
        for j in range(nb_nghbr):
            n0.append(M_new[1+j][1])
        
        #fonction pour modifier les neighbors ELISE 
    
    return; 
'''