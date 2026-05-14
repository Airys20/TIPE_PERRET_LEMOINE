import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize 
from scipy.ndimage import binary_opening 
from orientation import fun_orientation  


#. Pretraitement de base

def niv_de_gris(path):

    img_nivgris = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img_nivgris  is not None: 
        return img_nivgris


"""
normilise :imposer  moyenne et variance a atteindre ici M0 et VAR0 

formule que l'on veut traduire  (cf article)
$$
\
G(i,j) =
\begin{cases}
M_0 + \sqrt{\dfrac{VAR_0 \, (I(i,j) - M)^2}{VAR}}, & \text{si } I(i,j) > M \\[1.2em]
M_0 - \sqrt{\dfrac{VAR_0 \, (I(i,j) - M)^2}{VAR}}, & \text{sinon.}
\end{cases}
\
$$
$$
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

    if VAR < 1e-9: #on rempli tt pour eviter la div par 0 
        G = np.full_like(I, fill_value=M0, dtype=np.float64)  
    else:
        
        d = I - M
        ajustement = np.sqrt((VAR0 * (d**2)) / VAR) 
        #pour chaque pixel : écart normalisé 
        #conserve l'amplitude relative p/r à la moyenne, 
        # mais à échelle de la variance cible VAR0

        G = np.where(I>M, M0+ ajustement, M0 - ajustement ) 
        # construit img normalisée pixel par pixel :
        #   plus clair que M  → M0 + ajustement 
        #   plus sombre que M → M0 - ajustement 
        #-> signe conservé, amplitude remise vers VAR0
    return np.clip(G, 0, 255).astype(np.uint8) 
    #recadre entre [0,255 ] et passe format uint8 sinon bug



def fill_holes_safe(binary255): 
    """
     remplis les trous du masque pour qu'il soit uniforme 
    """
    m = binary255.copy()
    h, w = m.shape

    # force bordure noire pour demarrer floodfill dans fond
    m[0, :] = 0
    m[-1, :] = 0
    m[:, 0] = 0
    m[:, -1] = 0

    flood = m.copy()
    mask_ff = np.zeros((h + 2, w + 2), np.uint8) #zone de "coloriage"

    # floodfill depuis (0,0) qui est maintenant garanti fond (0)
    cv2.floodFill(flood, mask_ff, (0, 0), 255)

    # trous = zones non atteintes par floodfill
    holes = cv2.bitwise_not(flood) # fond=0, empreinte=0, trous=255 
    filled = cv2.bitwise_or(m, holes) # fusionne binary255 et trous 
    return filled


def masque_fun_v3(img_grise, debug=False):
    """
    Masque ROI :
    0 = fond
    255 = empreinte
    """
    # 1) flou fort pour effacer crêtes 
    flou_gros = cv2.GaussianBlur(img_grise, (0, 0), 25.0)

    # 2) Otsu: trouve meilleur seuil pour passage N&B
    thr, mask = cv2.threshold(flou_gros, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 3) décider inversion en regardant les coins 
    corners = np.array([mask[0,0], mask[0,-1], mask[-1,0], mask[-1,-1]])
    if np.mean(corners) > 127:
        mask = 255 - mask

    # 4) fermeture morpho (bouche trous)
    kclose = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kclose, iterations=2)

    # 5) plus grande comp connexe
    num, lbl, stats, _ = cv2.connectedComponentsWithStats(\
        (mask > 0).astype(np.uint8), 8)
    #num = nbr de composantes trouvées
    #lbl = image de même taille que mask, pour chaque pixel : numéro de sa comp connexe 
    #stats = tableau de stats pour chaque comp connexe : [x_min, y_min, largeur, hauteur, aire]

    if num > 1:
        #aire la + grd(fond exclus)
        max_comp_connexe = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA]) 
        #creer masque booleen
        mask = (lbl == max_comp_connexe).astype(np.uint8) * 255 

    # 6) remplissage trous
    mask = fill_holes_safe(mask)

   
    # 7) dilatation -> éviter masque trop “serré”
    kdil = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask = cv2.dilate(mask, kdil, iterations=1)

    # 8) lissage bords 
    kopen = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kopen, iterations=1)

    return mask



#. code principal

def pretraitements(filename):

    #nom du fichier entree+ sortie 
    output_filename = 'minutiae_detection\pretraitees\empreinte4_pretraitee.jpg'
    mask_output_filename ='minutiae_detection/masque/masque.png'

    #recup image
    image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE) 
    # resize pr meilleur "generalisation" 

    image = cv2.resize(image, (512, 512))  


    #TRANSFORMATION DE L'IMAGE

    #2) normalisation  
    tab_normal = normalise_fun(image)
    
    #3)masque ROI
    masque = masque_fun_v3(tab_normal)
    cv2.imwrite(mask_output_filename, masque)

    
    #4) orientation locale
    W_BLOCK = 16
    O_bloque = fun_orientation(tab_normal, masque=masque, w=W_BLOCK)
    np.save('minutiae_detection/output_orientation/O_bloque.npy', O_bloque)
    
    
    #5): CLAHE (contraste adaptatif)
    #  améliore contraste local 
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))

    '''
    CLAHE = Contrast Limited Adaptive Histogram Equalization
    augmente contraste dans chaque petite zone -> bien pour zone sur ou sous exposee de l'img
    paramètre : clipLimit=2.5 : Limite le renforcement du contraste
    tileGridSize=(8, 8) : découpe img en zones  de 8x8 blocs 
    '''
    contrast = clahe.apply(tab_normal) #applique le contraste 

    #6) lisser
    filtered = cv2.bilateralFilter(contrast, 5, 100, 125) #lisse img mais garde contours
    '''
    9=taille filtre autour de chaque point 
    100=+élevé + peut mélanger des tons différents
    100	= +élevé + regarde loin autour du pixel
    '''


    # 7)"binarisation" passe de gris a N&B mais localement -> bien pour zones sur/ss exposées
    binary = cv2.adaptiveThreshold(filtered, 255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV, blockSize=21, C=10)
    '''
    255=val pr pixel blanc
    blockSize = taillce zone traite localement
    cv2.THRESH_BINARY_INV : les zones sombres deviennent blanche -> pour meilleur lecture apres
    C ≈ abbaisse seuil pour etre detecte comme noir -> evite les faux positif/bruit
    '''

    # 8)morph
    # garde que les lignes vraiment présentes, sans petits bouts ni trous 
    morph = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) #supprime pt blancs
    morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)) #remplis les trous noirs dans les lignes blanches

    morph_bin = (morph > 0).astype(np.uint8)#transforme img en 0et1
    cleaned= binary_opening(morph_bin, structure=np.ones((3, 3))) #comme morph_open mais plus math -> mieux pour squeletisation


    #9) squelettisation
   
    skeleton = skeletonize(cleaned) # -> ensemble de 0 et 1 
    # conversion en image enregistrable 
    skeleton_img = (skeleton * 255).astype(np.uint8)
    inverted = cv2.bitwise_not(skeleton_img) #inverse couleur -> facilite traitement  

    # 10) sauvegadre
    cv2.imwrite(output_filename, inverted)
    print("Image prétraitée enregistree ")


    #------------------------------------------------------------------
    #AFFICHAGE
    
    
    plt.subplot(1, 4, 1)
    plt.imshow(image, cmap='gray')
    plt.title("originale")
    plt.axis('off')

    plt.subplot(1, 4, 2)
    plt.title("après Gabor")
    plt.axis('off')

    plt.subplot(1, 4, 3)
    plt.imshow(morph, cmap='gray')
    plt.title("après clean")
    plt.axis('off')

    plt.subplot(1, 4, 4)
    plt.imshow(inverted, cmap='gray')
    plt.title("squelette")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    return output_filename
