import numpy as np
import json

''' *************************************** OUTILS *************************************** '''

def wrap_angle(theta):  #normalisation de l'angle theta
    return (theta + np.pi) % (2*np.pi) - np.pi

def angle_diff_biom(theta1, theta2):   #calcul de la différence entre nos deux angles une fois normalisés
    d = abs(wrap_angle(theta1 - theta2))
    return min(d, abs(d - np.pi))

SPATIAL_TOL = 0.2   # tolérance spatiale (≈ ±10 px)
ANGLE_TOL = 0.100      # tolérance angle (≈ ±10°)

''' *************************************** NEIGHBORS *************************************** '''

_neighbors_cache = {}

def chgt_base(M,t): 
    xm, ym, theta_m = M[t]
    M_new = []
    for i in range(len(M)):
        xi, yi, theta_i = M[i]
        xi -= xm 
        yi -= ym 
        xtmp = xi
        xi =  np.cos(theta_m)*xi + np.sin(theta_m)*yi
        yi = -np.sin(theta_m)*xtmp + np.cos(theta_m)*yi
        theta_new = wrap_angle(theta_i - theta_m)
        M_new.append((xi,yi,theta_new))        
    return(M_new)

def fusion(L1,L2): #fonction auxiliaire pour mon tri fusion 
    n1,n2=len(L1),len(L2)
    L12=[0]*(n1+n2);i1=i2=i=0
    while i1<n1 and i2<n2:
        if L1[i1][0] < L2[i2][0]:
            L12[i]=L1[i1];i1+=1
        else:
            L12[i]=L2[i2];i2+=1
        i+=1
    while i1<n1: L12[i]=L1[i1];i1+=1;i+=1
    while i2<n2: L12[i]=L2[i2];i2+=1;i+=1
    return L12

def tri_fusion_recursif(L):  
    n=len(L)
    if n>1:
        p=n//2
        L1=L[:p];L2=L[p:]
        tri_fusion_recursif(L1)
        tri_fusion_recursif(L2)
        L[:] = fusion(L1,L2)

def distances(M):
    M_new=[(M[i][1],i) for i in range(len(M))]
    tri_fusion_recursif(M_new)
    return M_new

def neighbors(M , nb_nghbr, i):
    key=(id(M),i)
    if key in _neighbors_cache: return _neighbors_cache[key]
    M_tmp=chgt_base(M,i)
    M_new=distances(M_tmp)
    n0=[]
    j=1
    while len(n0)<nb_nghbr and j<len(M_new):
        n0.append(M_new[j][1]); j+=1
    _neighbors_cache[key]=n0
    return n0

''' *************************************** SUBSTRUCTURES *************************************** '''

lim_theta = 0.5
nb_nghbr = 15

def create_substructures(M, n1):
    substruct=[]
    for i in range(len(n1)):
        x,y,_=M[n1[i]]
        substruct.append([x,y])
    return(substruct)

def corresponding_substruct(substruct_c, substruct_r):
    if len(substruct_c)!=len(substruct_r): return False
    tol=5
    for i in range(len(substruct_c)):
        xc,yc=substruct_c[i]; xr,yr=substruct_r[i]
        if abs(xc-xr)>tol or abs(yc-yr)>tol: return False
    return True

''' *************************************** ALIGNEMENT *************************************** '''

def kabsch_umeyama(A,B): #fonction qui permet de calculer les paramètres de transformation rigide
    assert A.shape==B.shape
    EA=np.mean(A,axis=0);EB=np.mean(B,axis=0)
    VarA=np.mean(np.linalg.norm(A-EA,axis=1)**2)
    H=((A-EA).T @ (B-EB))/A.shape[0]
    U,D,VT=np.linalg.svd(H)
    d=np.sign(np.linalg.det(U)*np.linalg.det(VT))
    S=np.diag([1]*(A.shape[1]-1)+[d])
    R=U@S@VT
    c=VarA/np.trace(np.diag(D)@S)
    t=EA - c*R@EB
    return R,c,t

def apply_transformation(minutiae,R,c,t):
    new_minutiae=[]
    angle_rot=np.arctan2(R[1,0],R[0,0])
    for m in minutiae:
        x,y=m["coordonnees"]; theta=m["orientation"]
        xy_new=t + c*(R@np.array([x,y]))
        theta_new=wrap_angle(theta + angle_rot)
        new_minutiae.append({"coordonnees":xy_new.tolist(),"orientation":theta_new})
    return new_minutiae

''' *************************************** MATCHING *************************************** '''

def classify(minutiae_recherche,minutiae_catalogue):
    nr=len(minutiae_recherche)
    nc=len(minutiae_catalogue)
    paired=[0]*nc
    for i in range(nc):
        xc,yc=minutiae_catalogue[i]["coordonnees"]
        dc=minutiae_catalogue[i]["orientation"]
        best=0
        for j in range(nr):
            xr,yr=minutiae_recherche[j]["coordonnees"]
            dr=minutiae_recherche[j]["orientation"]
            if abs(xr-xc)<SPATIAL_TOL and abs(yr-yc)<SPATIAL_TOL:
                if angle_diff_biom(dr,dc)<ANGLE_TOL:
                    best=2; break
                else:
                    best=1
        paired[i]=best
    return paired

def matching_score(minutiae_catalogue,paired):
    nc=len(minutiae_catalogue)
    Npair=paired.count(1);
    Nmatch=paired.count(2)
    return ((Nmatch + 0.5*Npair)/nc)*100

''' *************************************** MATCHING GLOBAL *************************************** '''

def global_matching_jips(data_catalogue,data_recherche):
    
    with open(data_catalogue,"r") as f: catalogue=json.load(f)
    with open(data_recherche,"r") as f: recherche=json.load(f)
    
    M_recherche=[(m["coordonnees"][0],m["coordonnees"][1],m["orientation"])
                 for m in recherche[0]["minutiae"]]
    best_score=0
    best_name=None
    
    for p in range(len(catalogue)):
        print("Analyse de : " + catalogue[p]["nom"])
        M_catalogue=[(m["coordonnees"][0],m["coordonnees"][1],m["orientation"])
                     for m in catalogue[p]["minutiae"]]
        best_score = 0 
        
        for i in range(len(M_recherche)):
            for j in range(len(M_catalogue)):
                if abs(wrap_angle(M_recherche[i][2]-M_catalogue[j][2]))>lim_theta:
                    continue
                
                sub_r=create_substructures(M_recherche,neighbors(M_recherche,nb_nghbr,i))
                sub_c=create_substructures(M_catalogue,neighbors(M_catalogue,nb_nghbr,j))
                
                if not corresponding_substruct(sub_c,sub_r): continue
                
                R,c,t=kabsch_umeyama(np.array(sub_r),np.array(sub_c))
                minutiae_aligned=apply_transformation(recherche[0]["minutiae"],R,c,t)
                paired=classify(minutiae_aligned,catalogue[p]["minutiae"])
                score=matching_score(catalogue[p]["minutiae"],paired)
                 
                
                
                if score>best_score:
                    best_score=score
                    best_name=catalogue[p]["nom"]
                    print("best score :" , best_score, "i: ", i)
                    
                if (j>50 and best_score<60) :
                    break
                if best_score > 90 : 
                    break
            if (i>30 and best_score<70) :
                    break
            '''if (i % 10) == 0 : 
                    print(i) '''
        print("Name :", catalogue[p]["nom"],  "Best_score:", best_score)
    return best_name,best_score

''' *************************************** TEST *************************************** '''
nom, score = global_matching_jips("catalogue.json", "recherche.json")
print("\nMeilleure correspondance :", nom)
print("Score :", score)