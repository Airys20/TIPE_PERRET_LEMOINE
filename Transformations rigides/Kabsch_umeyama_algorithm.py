import json_utils as ju
import numpy as np 
import shutil
import selection_des_minuties as am




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

def transformation_rigide(catalogue, recherche):
    Mr = ju.get_data(0, recherche)["minutiae"]
    Mr_list = []
    for j in range(len(Mr)):
        (x,y) = Mr[j]['coordonnees']
        Mr_list.append([x,y])
    Mr_matrix = am.distance(np.array(Mr_list))
        
    base_c = ju._charger_base(catalogue)
    n = len(base_c)
   
    #nouveau catalogue json 
    shutil.copy(catalogue, 'new_catalogue.json')
         
    for i in range(n): 
        Mc = ju.get_data(i, catalogue)["minutiae"]
        Mc_list = []
        for j in range(len(Mc)):
            (x,y) = Mc[j]['coordonnees']
            Mc_list.append([x,y])
        Mc_matrix = am.distance( np.array(Mc_list))
        
        (R,c,t) = kabsch_umeyama(Mr_matrix, Mc_matrix)
        B = np.array([t + c * R @ b for b in Mc_matrix])
          
        
        #modification des coordonnées b 
        def f(m) :
            return ([t + c * R @ m][0].tolist())
        
    
        ju.iter_coordonnees ('new_catalogue.json', f, i)
        
    return()

transformation_rigide('catalogue.json', 'recherche.json')
        
        
        