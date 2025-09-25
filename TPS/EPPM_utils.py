import numpy as np 


def polar_coordinates(M0, M1, ui, vj):
     
    new_M0 = []
    new_M1 = []
     
    n = len(M1)
    nc = len(M0)
    (x,y) = ui 
    (x3, y3) = vj
    
    for i in range(nc):
        (x2,y2) = M0[i]
        if (x == x2 or y==y2):
            new_M1.append((0,0))
        else:
            new_M0.append((np.sqrt((x-x2)**2 + (y-y2)**2), np.arctan(np.sqrt(((y-y2)/(x-x2))**2))))
        
    for i in range(n) : 
        (x4,y4) = M1[i]
        if (x3 == x4 or y3==y4):
            new_M1.append((0,0))
        else:
            new_M1.append((np.sqrt((x3-x4)**2 + (y3-y4)**2), np.arctan(np.sqrt(((y3-y4)/(x3-x4))**2))))
        
    return (new_M0, new_M1)


M0 = [(1,1), (2,2), (3,3)]
print(polar_coordinates(M0, M0, (1,1), (1,1)))



def classify(M0, M1):   
        
    n =  len(M1)
    nc = len(M0)
    matched = 0
   
    nmin = min(n, nc)
    
    for i in range(nc) :
        (x,y) = M0[i]
        for j in range(n) : 
            (x1, y1) = M1[i]       
            if (x1> x-1) and x1<(x+1) and y1<(y+1) and y1>(y-1) :
               matched = matched + 1 
              
              
    return(matched)



def MPj_matrix(M1, M0, ui):
    MPj = []
    
    np = len(M1)
    nc = len(M0)
    
    for i in range(np):
        (new_M0, new_M1) = polar_coordinates(M0, M1, ui, M1[i])
        MPj.append( classify(new_M0, new_M1) ) 
        
    return MPj

print( MPj_matrix(M0, M0, (1,1))) #-> résultat non logique 
        

def decalages(ui, vj, new_ui, new_vj):
    (xu, yu) = ui
    (xv, yv) = vj
    (_ , Ou) = new_ui
    (_ , Ov) = new_vj
    
    delta_x = np.abs( xu - xv ) #décalages transationnels 
    delta_y = np.abs( yu - yv ) 
    delta_O = np.abs( Ou - Ov ) #décalage rotationel 
    
    return(deltax, deltay, deltaO)





def main(): 
    #loop through all possible ui and vj and collect votes 
        #Extract minitiae
        
        #Pick a reference pair and move to polar coordonates 
        
        #Rotate the query set and find candidate matches 
        
        #Compute the transform 
        
        #Apply the transform to every other minutiae from the query set and vote 
        
    #Bin and pick the top candidates 
    
    # select the best global alignment 
     


'''def ...(M0, M1):
    dico_personne = (empreinte[0]["minutiae"])
    dico_catalogue = (catalogue[personne]["minutiae"])

    return '''