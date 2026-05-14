import cv2
import numpy as np
import matplotlib.pyplot as plt
import math


def filtrer_minuties_proches_bord(minuties, mask01, marge_px=12,/
                                   img_size=512):
    """
    Supprime minuties trop proches du bord de la ROI 
    marge_px : distance minimale
    """
    if mask01 is None:
        return minuties  # rien à faire

    # cv2.distanceTransform -> 0=background, >0=foreground
    mask_u8 = (mask01.astype(np.uint8) * 255)

    # pour chaque pix blanc de masque calc dist au ROI 
    dist = cv2.distanceTransform(mask_u8, distanceType=cv2.DIST_L2,/
                                  maskSize=3)

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

def find_minuatiae(filename, output_filename,/
                    mask_filename=None, mask=None):
    
    
    def find(squelette, mask01=None):

        minutiae_ending = set()
        minutiae_bifurcation = set()
        points = []

        rows, cols = squelette.shape #recupere taille squelette 

        for i in range(3, rows - 3):
            for j in range(3, cols - 3):
                if mask01 is not None and mask01[i, j] == 0:
                    continue # si pas dans masque, on check pas
                if squelette[i][j] == 1:
                    #on verifie couleurs des 8 voisins
                    voisins = squelette[i-1:i+2, j-1:j+2] 
                    #enleve pixel centre
                    cmpt = int(np.sum(voisins)) - 1 
                    point = (j, i)

                    if cmpt == 1:
                        voisins_5 = squelette[i-2:i+3, j-2:j+3]
                        if int(np.sum(voisins_5)) > 3:
                            continue
                        minutiae_ending.add(point)
                        points.append(point)
                    
                    elif cmpt == 3:
                        minutiae_bifurcation.add(point)
                        points.append(point)

        res = []
        for (x, y) in points:
            if (x, y) in minutiae_ending:
                typ = "ending"
            elif (x, y) in minutiae_bifurcation:
                typ = "bifurcation"
            else:
                continue  # n'arrive pas
            
            # [0,0] = orient_faux placeholder
            res.append([[x/512, y/512], typ, [0, 0]])  
        

        print("endings:", len(minutiae_ending))
        print("bifurcations:", len(minutiae_bifurcation))  
        print("total points:", len(points))
        print("res final:", len(res))
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

    #passse en binaire pour traitement
    _, binaire = cv2.threshold(img, 127, 1, cv2.THRESH_BINARY_INV)

    # minutiae[[coord], type, [dx, dy]]
    minutiae= find(binaire, mask01=mask01)




    def filtrer_minuties_trop_proches(minuties, /
                                       min_dist_px=20, img_size=512):
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
            # on compare aux points déjà gardés 
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

    minut = filtrer_minuties_trop_proches(minutiae,\
                                           min_dist_px=6, img_size=512)
    minut = filtrer_minuties_proches_bord(minut, mask01,\
                                           marge_px=12, img_size=512)
    # passe en couleur pour dessin des ronds
    color_image = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # affichage ronds+ligne

    for [x_n, y_n], typ, orient in minut:
        x = int(round(x_n * 512))
        y = int(round(y_n * 512))

        if typ == "ending":
            color = (0, 0, 255)  # rouge
        elif typ == "bifurcation":
            color = (0, 255, 0)  # vert
        else:
            color = (0, 255, 255)  # jaune pour "ERREUR"

        # rond
        
        cv2.circle(color_image, (x, y), 3, color, 1)
   
    cv2.imwrite(output_filename, color_image)

    plt.figure(figsize=(10, 10))
    plt.imshow(color_image[..., ::-1])# inverse les canaux BGR → RGB pour matplotlib 
    plt.title("minutiae detecteees")
    plt.axis("off")
    plt.show()
    return minut

