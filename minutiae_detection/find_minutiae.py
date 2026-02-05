import cv2
import numpy as np
import matplotlib.pyplot as plt
import math


def filtrer_minuties_proches_bord(minuties, mask01, marge_px=12, img_size=512):
    """
    Supprime les minuties trop proches du bord de la ROI (mask01).
    marge_px : distance minimale (en pixels) au bord du masque.
    """
    if mask01 is None:
        return minuties  # rien à faire

    # cv2.distanceTransform attend une image 8-bit avec 0=background, >0=foreground
    mask_u8 = (mask01.astype(np.uint8) * 255)

    # Distance (en pixels) au plus proche pixel 0 (donc au bord du masque)
    dist = cv2.distanceTransform(mask_u8, distanceType=cv2.DIST_L2, maskSize=3)

    keep = []
    for m in minuties:
        (x_n, y_n) = m[0]
        x = int(round(x_n * (img_size - 1)))
        y = int(round(y_n * (img_size - 1)))

        # si hors masque ou trop près du bord -> on enlève
        if mask01[y, x] == 0:
            continue
        if dist[y, x] < marge_px:
            continue

        keep.append(m)

    return keep

def find_minuatiae(filename, output_filename, mask_filename=None, mask=None):

    def find(squelette, mask01=None, sigma=0.5):

        minutiae_ending = []#stock minutiae
        minutiae_bifurcation = []
        minutiae_orientation = []

        rows, cols = squelette.shape #recupere nbr lignes et colonnes de l’img
        directions = [(-1,-1), (-1,0), (-1,1),
                    (0,-1),          (0,1),
                    (1,-1),  (1,0),  (1,1)] # pixels a checker quand on a celui du milieu [0,0]

    #on regarde chaque picel de l'img
        for i in range(3, rows - 3):  # on évite les bords sinon bugs (bc n'a pas de voisinsins)
            for j in range(3, cols - 3):
                if mask01 is not None and mask01[i, j] == 0: # ATTENTION SI PAS DANS MAQSQUE ZAPPER 
                    continue

                elif squelette[i][j] == 1: # si est blanc
                    voisins = squelette[i-1:i+2, j-1:j+2]
                    cmpt = np.sum(voisins) - 1  # suppr [i][j]

                    point = (j, i)  # (x, y), inverse bc img = matrice , inverse des axes normaux

                    # id minutiae
                    if cmpt==1:
                        minutiae_ending.append(point) #1 voisin blanc = ending
                    elif cmpt==3:
                        minutiae_bifurcation.append(point)#3 voisin blanc ->biffurcassion
                    else:

                        continue  # pas  minutiae

                    #cherche d'orientation sur 4 puis 3 puis 2px pour que tt les points en est unee mm si pas hyper precise
                    angle_bool_trouve = False

                    #  4 pixels
                    for dy, dx in directions:
                        try:
                            if (squelette[i+ dy][j +dx] == 1 and
                                squelette[i+2*dy][j+ 2*dx] == 1 and
                                squelette[i+ 3*dy][j+3*dx] == 1):

                                angle = math.atan2(dy, dx) #angle entre vect et axe x

                                minutiae_orientation.append((point, angle))
                                angle_bool_trouve = True #pas besoin chercher les autre bc trouvé le max
                                break

                        except IndexError:
                            continue #⚠️⚠️eviter les bords!!!!!!

                    # 3px (mm chose)
                    if not angle_bool_trouve:
                        for dy, dx in directions:
                            try:
                                if (squelette[i + dy][j + dx] == 1 and
                                    squelette[i + 2*dy][j + 2*dx] == 1):
                                    angle = math.atan2(dy, dx)
                                    minutiae_orientation.append((point, angle))
                                    angle_bool_trouve = True
                                    break
                            except IndexError:
                                continue

                    # 2px (pareil)
                    if not angle_bool_trouve:
                        for dy, dx in directions:
                            try:
                                if squelette[i + dy][j + dx] == 1:
                                    angle = math.atan2(dy, dx)
                                    minutiae_orientation.append((point, angle))
                                    break
                            except IndexError:
                                continue
        
        res = [] #prepare tab adapté au style de struc, cf com de la fonction add personne dans json_utils
    
        for (x, y), angle in minutiae_orientation:
            if (x, y) in minutiae_ending:
                typ = "ending"
            elif (x, y) in minutiae_bifurcation:
                typ = "bifurcation"
            else:
                typ = "ERREUR" #PAS CENSE ARRIVER

            #trouve comp sur x et y pour tracer trzait d'oriebntation
            dy = round(math.sin(angle), 4)

            res.append([[x/512, y/512], typ, [dx, dy]])  




        return res


   # import+ passage binaire
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(filename)

    if mask is None and mask_filename is not None:
        mask = cv2.imread(mask_filename, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (512, 512))
    if mask is not None:
        # force masque binaire 0/1 pour simplifier
        mask01 = (mask > 0).astype(np.uint8)
    else:
        mask01 = None

        
    _, binaire = cv2.threshold(img, 127, 1, cv2.THRESH_BINARY_INV)#passse en binaire pour traitement

    # minutiae[[coord], type, [dx, dy]]
    minutiae= find(binaire, mask01=mask01)




    def filtrer_minuties_trop_proches(minuties,  min_dist_px=20, img_size=512):
        """
        liste minut: [[x_n, y_n], typ, orient]  OU [[x_n, y_n], typ, orient, ]
        min_dist_px: distance min en px entre 2 minut (seuil avant enlever)
        

        renvoie: liste sans doublons
        """
        # convertit en (x_px, y_px) -> on retrouve les coord init
        pts = []
        for m in minuties:
            (x_n, y_n) = m[0]
            x= int(round(x_n *(img_size- 1)))
            y =int(round(y_n *(img_size- 1)))
            pts.append((x, y, m))

        
        #pts.sort(key=lambda t: (t[1], t[0]))  # y puis x

        keep = []
        keep_xy = []
        d2_min = min_dist_px * min_dist_px

        for x, y, m in pts:
            ok = True
            # on compare aux points déjà gardés complexité ignoble mais OK si pas trop)
            for (xk, yk) in keep_xy:
                dx = x - xk
                dy = y - yk
                if dx*dx + dy*dy < d2_min:
                    ok = False
                    break
            if ok:
                keep.append(m)
                keep_xy.append((x, y))

        return keep

    minutt = filtrer_minuties_trop_proches(minutiae, min_dist_px=6, img_size=512)
    minutt = filtrer_minuties_proches_bord(minutt, mask01, marge_px=12, img_size=512)
    # passe en couleur pour dessin des ronds
    color_image = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # affichageronds+ligne

    for [x_n, y_n], typ, [dx, dy] in minutt:
        x = int(round(x_n * 512))
        y = int(round(y_n * 512))

        if typ == "ending":
            color = (0, 0, 255)  # rouge
        elif typ == "bifurcation":
            color = (0, 255, 0)  # vert
        else:
            color = (0, 255, 255)  # jaune pour "ERREUR"

        # rond
        
        cv2.circle(color_image, (x, y), 1, color, 1)
    '''
        # trait orientartion
        lx = int(round(x + dx * 6)) #calc point arrivée trait orient. apres 6px
        ly = int(round(y + dy * 6))
        cv2.line(color_image, (x, y), (lx, ly), (255, 0, 0), 1)
        '''
    cv2.imwrite(output_filename, color_image)

    plt.figure(figsize=(10, 10))
    plt.imshow(color_image[..., ::-1])# inversant les canaux BGR → RGB pour matplotlib ??
    plt.title("minutiae detecteees")
    plt.axis("off")
    plt.show()
    return minutt

