import numpy as np 
#trouver le centre de l'empreinte et prendre les 50/100 minuties les plus proches

def center(Matrix):
    x = np.mean(M, axis=0)
    y = np.mean(M, axis=1)
    return(x,y)

def new_cat(M): # avec i : l'emplacement de la minutie de réf dans M ; M : l'ensemble des coordonnées de l'empreinte  
    (x0,y0)   =  center(M)
    distances = [] 
    n = len(M)
    j = 0
    while (j< (n-1)) :
        (x,y) = M[j]
        if(x==x0 and y==y0):
            distances.append(0)
        else:
            distances.append(np.sqrt((x0-x)**2 + (y0-y)**2))
        j = j+1    
    temp = sorted(distances)
    dist_sort = []
    for k in range(len(temp)):
        dist_sort.append(distances.index(temp[k]))
        
    new_list = []   
    for j in range(500):
        m = M[dist_sort[j]]
        new_list.append(m)
        new_matrix = np.array(new_list)
        
    return (new_matrix)
    