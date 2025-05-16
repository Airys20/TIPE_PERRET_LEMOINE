import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

def find_minuatiae(filename, output_filename):

    def find(squelette):

        minutiae_ending = []#stock minutiae
        minutiae_bifurcation = []
        minutiae_orientation = []

        rows, cols = squelette.shape #recupere nbr lignes et colonnes de l’img
        directions = [(-1,-1), (-1,0), (-1,1),
                    (0,-1),          (0,1),
                    (1,-1),  (1,0),  (1,1)] # pixels a checker quand on a celui du milieu [0,0]

    #on regarde chaque picel de l'img
        for i in range(3, rows - 3):  # on évite les bords sinon bugs (bc n'a pas de voisins)
            for j in range(3, cols - 3):

                if squelette[i][j] == 1: # si est blanc
                    vois = squelette[i-1:i+2, j-1:j+2]
                    cmpt = np.sum(vois) - 1  # suppr [i][j]

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
                            if (squelette[i + dy][j + dx] == 1 and
                                squelette[i + 2*dy][j + 2*dx] == 1 and
                                squelette[i + 3*dy][j + 3*dx] == 1):

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

            res.append([[x, y], typ, [dx, dy]])

        return res


   # import+ passage binaire
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    _, binaire = cv2.threshold(img, 127, 1, cv2.THRESH_BINARY_INV)#passse en binaire pour traitement

    # minutiae[[coord], type, [dx, dy]]
    minutiae = find(binaire)

    # passe en couleur pour dessin des ronds
    color_image = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # affichageronds+ligne

    for [x, y], typ, [dx, dy] in minutiae:
        if typ == "ending":
            color = (0, 0, 255)  # rouge
        elif typ == "bifurcation":
            color = (0, 255, 0)  # vert
        else:
            color = (0, 255, 255)  # jaune pour "ERREUR"

        # rond
        
        cv2.circle(color_image, (x, y), 3, color, 1)

        # trait orientartion
        lx = int(round(x + dx * 6)) #calc point arrivée trait orient. apres 6px
        ly = int(round(y + dy * 6))
        cv2.line(color_image, (x, y), (lx, ly), (255, 0, 0), 1)
        
    cv2.imwrite(output_filename, color_image)

    plt.figure(figsize=(10, 10))
    plt.imshow(color_image[..., ::-1])# inversant les canaux BGR → RGB pour matplotlib ??
    plt.title("minutiae detecteees")
    plt.axis("off")
    plt.show()
    return minutiae
