# classify all the fingerprints 
    # find the number of paired and matched minutiaes 
        # paired = 2 minutiae fall into the same tolerance box 
        # matched = paired + same direction (with some tolerance)

import numpy 
import json
data_catalogue = 'catalogue.json'
data_empreinte = 'recherche.json'


def classify(personne):
    with open(data_catalogue, "r", encoding="utf-8") as f :
        catalogue = json.load(f)
    
    with open(data_empreinte, "r", encoding="utf-8") as f2 :
        empreinte = json.load(f2)    
        
    dico_personne = (empreinte[0]["minutiae"])
    dico_catalogue = (catalogue[personne]["minutiae"])

    n =  len(dico_personne)
    paired = [ 0 for t in range(n)]
    
    for i in range(n) :
        
        
        np = len(dico_personne)
        
        coord_cat = dico_personne[i]["coordonées"]
        coord_pers = dico_personne[i]["coordonées"]
        
        for j in range(np) : 
        
                    
            if (coord_pers[0]> (coord_cat[0] - 10) and coord_pers[0]<(coord_cat[0]+10) and coord_pers[1]<(coord_cat[1]+10) and coord_pers[1]>(coord_cat[1]-10)) :
                dir_cat = dico_personne[j]["orientation"]
                dir_pers = dico_personne[j]["orientation"]
                if (dir_pers[0]> (dir_cat[0] - 5) and dir_pers[0]<(dir_cat[0]+5) and dir_pers[1]<(dir_cat[1]+5) and dir_pers[1]>(dir_cat[1]-5)) :
                 #les mettre en matched 
                    paired[i]=1

                else :
                    paired[i]=2
        
        
            
    return(paired)


'''tableau paired : 0 si notpaired ; 2 si paired; 1 si matched'''           


# calcul the matching score 

def matching_score( personne, paired) :
    with open(data_catalogue, "r", encoding="utf-8") as f :
        catalogue = json.load(f)
    
    with open(data_empreinte, "r", encoding="utf-8") as f2 :
        empreinte = json.load(f2)    
        
    dico_personne = (empreinte[0]["minutiae"])
    dico_catalogue = (catalogue[personne]["minutiae"])
    
    np =  len(dico_personne)
    nc = len(dico_catalogue)
    
    Npair = 0 
    m = 0
    for i in range(np) :  
        if (paired[i] == 2 ):
            Npair = Npair + 1 
        elif (paired[i] == 1 ):
            m = m + 1
    
    if (Npair == 0) :
        return((100*m)/numpy.sqrt(nc*np))
    else : 
        return((100*m)/Npair)





#Calculer tous les matchings score et les ranger dans un tableau 

def global_matching(data_file):
    
    with open(data_file, "r", encoding="utf-8") as f :
        base = json.load(f)
    
    
    catalogue = (base)
    
    nc = len(catalogue)
    print(nc)
    i_max = 0;
    tab_score = [0 for i in range(nc)]
    for i in range(nc):
        paired = classify(i)
        tab_score[i] = matching_score( i, paired)
        print(tab_score[i])
        if abs((tab_score[i])-100) < abs(tab_score[i_max]-100):
            i_max = i
        

    nom = (base[i_max]["nom"])
    return(nom)
    
         
print(global_matching(data_catalogue) )  

# to get better result : match singular points 

