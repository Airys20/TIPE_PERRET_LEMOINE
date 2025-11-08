img = "minutiae_detection\input\empreinteS3.jpg" 


import numpy  as np
import cv2
import matplotlib.pyplot as plt



def niv_de_gris(path):

    img_nivgris = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img_nivgris  is not None: 
        return img_nivgris



def normalise_fun(img_grise, M0=100.0, VAR0=100.0):

    I = img_grise.astype(np.float64) #passe l'image en TABLEAU de la val de chaque pixel 
    
    if I.max() <= 1.0: 
        I *= 255.0 #on elargit les niv de gris 

    M = I.mean(); #val moy de gris
    VAR = I.var(); #variance moy de gris 

    if VAR < 1e-9: #cas si variance null on rempli tt pour eviter la div par 0 
        G = np.full_like(I, fill_value=M0, dtype=np.float64)  
    else:
        

        d = I - M
        ajustement = np.sqrt((VAR0 * (d**2)) / VAR) #=TABLEAU des parties sous la racine pour chaque pixel
        #(on conserve le signe p/r a la moyenne mais si + que M on le rend + que M0 et inv)
        G = np.where(I>M, M0+ ajustement, M0 - ajustement ) #CREER un nouv TABLEAU et rempli selon condition : np.where(condition, si sup a la moyenne, si inf a la moy)
    return np.clip(G, 0, 255).astype(np.uint8) #recadre entre [0,255 ] et repasse format uint8 ⚠️sinon bug



#. isolement empreinte (pas dans article mais bug sur orientation sinon)

"""
on veut crree un masque binaire pour isoler empreintre :
0= masque
255= empreinte
"""
def masque_fun(img_grise):
    flou = cv2.GaussianBlur(img_grise, (0,0), 3.0) 
    _, mask = cv2.threshold(flou, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU) #treshold avec valeur optimal deduite par l'algo renvoi val , masque 
    #on veut le masque en blanc ?
    if np.sum(mask==255) > np.sum(mask==0): #si + de noir que de blanc
        mask = 255 - mask #on inv
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15,15))  #COMMENT :⚠️si trop aggressif baissé ou augmenter la taille (memo 15 ok la plupart du temps)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k) #enleve petits trous
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k) #supprime petits points


    #PARTIE TROUVER DANS UN ARTICLE, marche mais jsp comment 
    num,lbl,stats,_ = cv2.connectedComponentsWithStats((mask>0).astype(np.uint8), 8)
    if num > 1:
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask = (lbl==largest).astype(np.uint8)*255
    


    mask = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)), 1) #lisse le tour
    return mask



gray = niv_de_gris(img)
tab_normal = normalise_fun(gray)
masque = masque_fun(tab_normal)
plt.figure(figsize=(10, 10))
plt.imshow(masque, cmap='Greys',  interpolation='nearest')# inversant les canaux BGR → RGB pour matplotlib ??
plt.title("masque ")
plt.axis("off")
plt.show()