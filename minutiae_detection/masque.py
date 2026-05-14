import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize 
from scipy.ndimage import binary_opening 




def masque_fun_v3(img_grise, debug=False):


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