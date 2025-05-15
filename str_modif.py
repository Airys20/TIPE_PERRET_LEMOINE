import numpy as np
import matplotlib.pyplot as plt

def str_modif (file, ajout):
    #trouver le format (regarder après le dernier point)
    n = len(file)
    i = n - 1
    l_file = list(file)
    s = [] 
    while (l_file[i] != '\\' and l_file[i] != '/') :
        i= i-1
    t = i 
    for t in range (i, n):
       s.append( l_file[i] )
       del l_file[i] 
    
    l_ajout = list(ajout)
    
    l_file.extend(l_ajout)
    l_file.extend(s)
    final = "".join(l_file)
    
    return final 



f = str_modif("Minutiae/empreinte8.jpg", "_pretraitement")   
print(f)


