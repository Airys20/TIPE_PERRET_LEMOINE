import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize 
from scipy.ndimage import binary_opening #supprime bruit
                                         #lisse les bords
                                         #ne touche pas ce qui esr bien formé
from skimage.measure import label, regionprops
from gabor import gabor_enhance # [ ] gabor.py doit etre dans le meme dossier
from orientation import fun_orientation  
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

    # ETAPE 2 : normalisation  ← AVANT TOUT LE RESTE
    #   (dans l'article c'est la toute première étape du pipeline)
    #   avant : cette étape était faite à la fin juste pour le masque,
    #           maintenant elle sert de base à toute la chaîne
    # ----------------------------------------------------------
    tab_normal = normalise_fun(image)
    # ----------------------------------------------------------
    # ETAPE 3 : masque ROI
    #   avant : calculé à la fin, après la squelettisation
    #           → le Gabor n'y avait pas accès
    #   maintenant : calculé ici pour être disponible pour Gabor
    # ----------------------------------------------------------
    masque = masque_fun_v3(tab_normal)
    cv2.imwrite('minutiae_detection/masque/masque.png', masque)

    # ----------------------------------------------------------
    # ETAPE 4 : orientation locale
    #   avant : orientation.py était appelé séparément dans main_find.py
    #           puis O_bloque était chargé depuis un .npy
    #   maintenant : on appelle fun_orientation directement ici
    #           pour que le Gabor puisse l'utiliser dans la même fonction
    # ----------------------------------------------------------
    #   W_BLOCK=16 doit correspondre à ce qu'utilise orientation.py
    W_BLOCK = 16
    O_bloque = fun_orientation(tab_normal, masque=masque, w=W_BLOCK)
    np.save('minutiae_detection/output_orientation/O_bloque.npy', O_bloque)
    #   [ ] on sauvegarde quand même le .npy pour que main_find.py
    #       puisse encore le recharger comme avant (pas de changement à faire là-bas)
 # ----------------------------------------------------------
    # ETAPE 5 : filtre de Gabor  ← NOUVEAU
    #   on améliore l'image normalisée AVANT la binarisation
    #   le Gabor renforce les crêtes et supprime le bruit orienté
    #   entrées : tab_normal (image), O_bloque (orientations), masque
    #   sortie  : enhanced  (même taille, même type uint8)
    # ----------------------------------------------------------
    enhanced = gabor_enhance(tab_normal, O_bloque, masque, w=W_BLOCK)

    # ----------------------------------------------------------
    # ETAPE 6 : CLAHE (contraste adaptatif)
    #   avant : appliqué sur `image` (brute)
    #   maintenant : appliqué sur `enhanced` (déjà améliorée par Gabor)
    #   les deux se complètent : Gabor améliore la structure,
    #   CLAHE améliore le contraste local résiduel
    # ----------------------------------------------------------
    #contraste

    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))

    '''
    CLAHE = Contrast Limited Adaptive Histogram Equalization
    augmente le contraste dans chaque petite zone -> bien pour zone sur ou sous exposee de l'img
    paramètre : clipLimit=2.5 : Limite le renforcement du contraste
    tileGridSize=(8, 8) : découpe img en zones  de 8x8 blocs 
    '''
    contrast = clahe.apply(enhanced) #applique le contraste 

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
    
    
   
    
    plt.subplot(1, 4, 1)
    plt.imshow(image, cmap='gray')
    plt.title("originale")
    plt.axis('off')

    plt.subplot(1, 4, 2)
    plt.imshow(enhanced, cmap='gray')   # ← nouveau panneau : résultat Gabor
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
