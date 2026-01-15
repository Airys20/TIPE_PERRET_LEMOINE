import numpy as np
import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
import json_utils as ju 



''' *************************************** NEIGHBORS *************************************** '''

def chgt_base(M,t):
    xm, ym, theta_m = M[t]
    M_new = []
    for i in range(len(M)):
        xi, yi = M[i][0:2]
        xi = xi - xm 
        yi = yi - ym 
        xtmp = xi 
                
        xi = np.cos(theta_m)*xi + np.sin(theta_m)
        yi = - np.sin(theta_m)*xtmp + np.cos(theta_m)
        
        M_new.append((xi,yi))        
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
        M_new.append((M[i][1], i))
    
    #tri fusion sur yi 
    tri_fusion_recursif(M_new)
    return M_new 

def neighbors(M , nb_nghbr, i):   
    
    M_tmp = chgt_base(M,i)
    M_new = distances(M_tmp)
        
    n0 = []
    for j in range(nb_nghbr):
        n0.append((M_new[1+j][1])) # on ajoute le numéro des minuties et non pas les coordonnées
    
    return (n0)
    
    
''' *************************************** SUBSTRUCTURES *************************************** '''
lim_theta = 0.5
nb_nghbr = 5

def corresponding_substruct(substruct_c, substruct_r): #PAS ENCORE TESTEE 
    bool = True
    
    j = 0
    for i in range(len(substruct_c)):
        while j < (len(substruct_c)):
            xi, yi = substruct_c[0:1][0]
            xj, yj = substruct_r[0:1][0]
            
            if(xi == xj) : 
                m = 0
            else :
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


def create_substructures(M, n1):
    substruct = []
    for i in range(len(n1)):
        x, y, _ = M[n1[i]]
        substruct.append([x,y])
    return(substruct)

def substructures(M_recherche, M_catalogue):
    b = False
    for i in range(len(M_recherche)):
        for j in range(len(M_catalogue)):      
            theta_r = M_recherche[i][2]
            theta_c = M_catalogue[j][2]
            
            if True: #(abs(theta_r - theta_c) < lim_theta)
                #the two minutiaes are regarded to be the corresponding minutiaes 
                substruct_c = create_substructures(M_catalogue, (neighbors(M_catalogue , nb_nghbr, j)))
                substruct_r = create_substructures(M_recherche, (neighbors(M_recherche , nb_nghbr, i)))
                
                if corresponding_substruct(substruct_c, substruct_r):
                    b = True
                    print(True)
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

def kabsch_umeyama(A, B):  # article : Aligning point patterns with Kabsch–Umeyama algorithm by Tuomas Siipola
    assert A.shape == B.shape
    n, m = A.shape
    print(A.shape)
    
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


def main_align(data_catalogue, data_recherche, i):
    base_catalogue = ju._charger_base(data_catalogue)
    base_recherche = ju._charger_base(data_recherche)
    
    #ju.clear_base("new_catalogue.json")
    
    l_nb_c = ju.compter_minuties_par_empreinte(data_catalogue)
    l_nb_r =  ju.compter_minuties_par_empreinte(data_recherche)
    
    dico_catalogue = base_catalogue[i]
    M_catalogue = []
    
    _,_,nc = l_nb_c[i]
    _,_,nr = l_nb_r[i]
   
    for j in range(nc):
        xi = dico_catalogue["minutiae"][j]["coordonnees"][0]
        yi = dico_catalogue["minutiae"][j]["coordonnees"][1]
        theta_i = dico_catalogue["minutiae"][j]["orientation"]
        M_catalogue.append((xi, yi, theta_i))
        
    dico_recherche = base_recherche[0]
    M_recherche = []
    for j in range(nr):
        xi = dico_recherche["minutiae"][j]["coordonnees"][0]
        yi = dico_recherche["minutiae"][j]["coordonnees"][1]
        theta_i = dico_recherche["minutiae"][j]["orientation"]
        M_recherche.append((xi, yi, theta_i))
    
    b, substruct_c, substruct_r = substructures(M_recherche, M_catalogue)
    
    # --- Calcul de la transformation qui aligne Mc sur Mr
    if b :
        R, c, t = kabsch_umeyama(np.array(substruct_r), np.array(substruct_c))
        # --- Application de la transformation aux coordonnées et orientations
        def f(m):
            return (t + c * R @ np.array(m)).tolist()
            
        ju.iter_coordonnees('recherche.json',f,0)
        
        # --- Mise à jour des orientations
        base_new = ju._charger_base("recherche.json")
        minutiae = base_new[i]["minutiae"]

        ''' for m in minutiae:
            orient = np.array(m["orientation"])
            m["orientation"] = (R @ orient).tolist()
        '''
        
        

    print("\n Transformation rigide terminée — catalogue sauvegardé dans \n")
    
    return()    
                 
''' *************************************** TEST *************************************** '''         
main_align('catalogue.json', 'recherche.json', 0 )
                
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