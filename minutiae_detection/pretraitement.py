import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize 
from scipy.ndimage import binary_opening #supprime bruit
                                         #lisse les bords
                                         #ne touche pas ce qui esr bien formé
from skimage.measure import label, regionprops

import os



#[x] faire en sorte que ça renvoie tableau avec les coordonnées 


#. Pretraitement de base


def niv_de_gris(path):

    img_nivgris = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    #img_resize = cv2.resize(img_nivgris, (512, 512))  # resize pr meilleur "generalisation" <- pas forcement le bon mot
    if img_nivgris  is not None: 
        return img_nivgris

"""
Le but de normilise est d'imposer une moyenne et variance a atteindre ici M0 et VAR0 

formule que l'on veut traduire  (cf article)
$$
\
G(i,j) =
\begin{cases}
M_0 + \sqrt{\dfrac{VAR_0 \, (I(i,j) - M)^2}{VAR}}, & \text{si } I(i,j) > M \\[1.2em]
M_0 - \sqrt{\dfrac{VAR_0 \, (I(i,j) - M)^2}{VAR}}, & \text{sinon.}
\end{cases}
\

\
M = \dfrac{1}{N} \sum_{i,j} I(i,j),
\qquad
VAR = \dfrac{1}{N} \sum_{i,j} \bigl(I(i,j) - M\bigr)^2
\
$$
"""



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


import numpy as np
import cv2

def fill_holes(binary255):
    """
    Remplit les trous à l'intérieur d'un masque binaire 0/255.
    """
    h, w = binary255.shape
    flood = binary255.copy()
    mask_ff = np.zeros((h+2, w+2), np.uint8)

    # floodfill depuis le bord (0,0) supposé être du fond
    cv2.floodFill(flood, mask_ff, (0, 0), 255)

    # les trous = pixels restés à 0 dans flood
    holes = cv2.bitwise_not(flood)
    filled = cv2.bitwise_or(binary255, holes)
    return filled


def masque_fun_v2(img_grise):
    """
    Masque ROI doigt (0 fond, 255 empreinte) robuste.
    """
    # 1) flou TRÈS fort pour supprimer les crêtes (ne garder que la "forme")
    # -> ajuste sigma selon taille image (ici ça marche bien sur ~1000px)
    flou_gros = cv2.GaussianBlur(img_grise, (0, 0), 25.0)

    # 2) Otsu sur image très lissée
    _, mask = cv2.threshold(flou_gros, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 3) forcer la zone doigt en blanc (255)
    # (on veut que le fond majoritaire soit noir)
    if np.sum(mask == 255) > np.sum(mask == 0):
        mask = 255 - mask

    # 4) fermeture morpho GRANDE pour coller et boucher les trous
    kclose = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kclose, iterations=2)

    # 5) garder la plus grande composante connexe
    num, lbl, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), 8)
    if num > 1:
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask = ((lbl == largest).astype(np.uint8) * 255)

    # 6) remplir les trous internes (important)
    mask = fill_holes(mask)

    # 7) dé-sélectiver un peu : dilatation légère + lissage bord
    kdil = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.dilate(mask, kdil, iterations=1)

    # 8) optionnel: petit open pour lisser les bords
    kopen = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kopen, iterations=1)

    return mask

def fill_holes_safe(binary255):
    """
    Remplit les trous d'un masque 0/255 de manière robuste.
    Astuce : on force une bordure noire pour garantir que le fond est connecté au bord.
    """
    m = binary255.copy()
    h, w = m.shape

    # force une bordure noire (très important)
    m[0, :] = 0
    m[-1, :] = 0
    m[:, 0] = 0
    m[:, -1] = 0

    flood = m.copy()
    mask_ff = np.zeros((h + 2, w + 2), np.uint8)

    # floodfill depuis (0,0) qui est maintenant garanti fond (0)
    cv2.floodFill(flood, mask_ff, (0, 0), 255)

    # trous = zones non atteintes par floodfill
    holes = cv2.bitwise_not(flood)
    filled = cv2.bitwise_or(m, holes)
    return filled


def masque_fun_v3(img_grise, debug=False):
    """
    Masque ROI doigt robuste :
    0 = fond
    255 = empreinte
    """
    # 1) flou fort pour effacer les crêtes (forme globale)
    flou_gros = cv2.GaussianBlur(img_grise, (0, 0), 25.0)

    # 2) Otsu
    thr, mask = cv2.threshold(flou_gros, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 3) décider l’inversion en regardant les coins (le fond est aux coins)
    corners = np.array([mask[0,0], mask[0,-1], mask[-1,0], mask[-1,-1]])
    # si la majorité des coins est blanche, alors le fond est blanc -> on inverse
    if np.mean(corners) > 127:
        mask = 255 - mask

    if debug:
        print("Otsu thr =", thr, "| white ratio after otsu =", np.mean(mask==255))

    # 4) fermeture morpho (colle les zones, bouche trous)
    kclose = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kclose, iterations=2)

    if debug:
        print("white ratio after close =", np.mean(mask==255))

    # 5) plus grande composante connexe
    num, lbl, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), 8)
    if num > 1:
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask = (lbl == largest).astype(np.uint8) * 255

    if debug:
        print("white ratio after CC =", np.mean(mask==255))

    # 6) remplissage trous (safe)
    mask = fill_holes_safe(mask)

    if debug:
        print("white ratio after fill holes =", np.mean(mask==255))

    # 7) dilatation légère (optionnelle) pour éviter masque trop “serré”
    kdil = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask = cv2.dilate(mask, kdil, iterations=1)

    if debug:
        print("white ratio after dilate =", np.mean(mask==255))

    # 8) lissage bords léger
    kopen = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kopen, iterations=1)

    if debug:
        print("white ratio final =", np.mean(mask==255))

    return mask

#. code principal





def pretraitements(filename):

    #nom du fichier entree+ sortie (voir si on peut automatiser)
    
    output_filename = 'minutiae_detection\pretraitees\empreinte4_pretraitee.jpg'
    #[ ] a changer avec str_modif(filename , _pretraitement)

    #recup image
    image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE) #teinte de gris
    image = cv2.resize(image, (512, 512))  # resize pr meilleur "generalisation" <- pas forcement le bon mot



    #TRANSFORMATION DE L'IMAGE

    #contraste
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))

    '''
    CLAHE = Contrast Limited Adaptive Histogram Equalization
    augmente le contraste dans chaque petite zone -> bien pour zone sur ou sous exposee de l'img
    paramètre : clipLimit=2.5 : Limite le renforcement du contraste
    tileGridSize=(8, 8) : découpe img en zones  de 8x8 blocs 
    '''
    contrast = clahe.apply(image) #applique le contraste 

    #lisser
    filtered = cv2.bilateralFilter(contrast, 5, 100, 125) #lisse img mais garde contours
    '''
    9=taille filtre autour de chaque point 
    100=+élevé + peut mélanger des tons différents
    100	= +élevé + regarde loin autour du pixel
    '''


    # "binarisation" passe de gris a noir et blanc mais localement comme pour le contraste -> bien pour zones sur/ss exposées
    binary = cv2.adaptiveThreshold(filtered, 255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV, blockSize=21, C=10)
    '''
    255=val pr pixel blanc
    blockSize = taillce zone traite localement
    cv2.THRESH_BINARY_INV : les zones sombres deviennent blanche -> pour meilleur lecture apres
    C ≈ abbaisse seuil pour etre detecte comme noir -> evite les faux positif/bruit
    '''

    # garde que les lignes vraiment présentes, sans petits bouts ni trous dedans
    morph = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) #supprime pt blancs
    morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)) #remplis les trous noirs dans les lignes blanches

    #pretraitement avant squeletisation
    morph_bin = (morph > 0).astype(np.uint8)#transforme img en 0et1
    cleaned= binary_opening(morph_bin, structure=np.ones((3, 3))) #comme morph_open (l54) mais plus math -> mieux pour squeletisation






    #squelettisation
   
    skeleton = skeletonize(cleaned) # -> ensemble de 0 et 1 donc par visualisable ?
   
    

    # conversion en image enregistrable 
    skeleton_img = (skeleton * 255).astype(np.uint8) #transforme les 0 et 1 en vrai picel couleur en multipliant par 255
    inverted = cv2.bitwise_not(skeleton_img) #inverse couleur pour faciliter traitement futur 

    # sauvegadre
    cv2.imwrite(output_filename, inverted)
    print("Image prétraitée enregistree ")


    #CREATION DU MASQUE 
    
    
    gray = niv_de_gris(filename)
    image = cv2.resize(gray, (512, 512))
    tab_normal = normalise_fun(image)
    

    masque = masque_fun_v3(tab_normal)
    cv2.imwrite('minutiae_detection/masque/masque.png',masque)
    
    #compte rendu
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(image, cmap='gray')
    plt.title("originale")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(morph, cmap='gray')
    plt.title("après clean")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(inverted, cmap='gray')
    plt.title("squelette")
    plt.axis('off')

    plt.tight_layout() #pour que tout soit bien sur l'img
    plt.show()
    return output_filename
