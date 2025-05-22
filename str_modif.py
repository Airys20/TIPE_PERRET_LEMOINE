import numpy as np
import matplotlib.pyplot as plt

def str_modif (file, ajout, dossier):
    #trouver le format (regarder après le dernier point)
    n = len(file)
    i = n - 1
    j = 0
    t = 0
    f = 0
    
    
    l_file = list(file)
    s1 = [] 
    s2=list(dossier)
    s3=[]
    s4=[ajout]
    s5=[]
    
    
    while l_file[i] != "." :
        i= i-1
    
    
    
    for j in range (i, n):
       s5.append( l_file[j] )
       
    t = i
    while l_file[t] != '\\' and l_file[t] != '/' :
        t= t-1
    
    for j in range (t, i):
       s3.append( l_file[j] )
       
    f = t - 1 
    while l_file[f] != '\\' and l_file[f] != '/' :
        f= f-1
    
    for j in range (0, f+1):
       s1.append( l_file[j] )
    

    s1.extend(s2)
    s1.extend(s3)
    s1.extend(s4)
    s1.extend(s5)
    
    final = "".join(s1)
    
    return final 


  
'''
f = str_modif("Minutiae/input/empreinte8.jpg", "_pretraitement", "output" )   
print(f)
'''
