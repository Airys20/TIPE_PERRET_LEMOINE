import cv2
import numpy as np
import matplotlib.pyplot as plt

#[ ]trouver l'orientation 
def find_minuatiae(filename,output_filename):

    def find(squelette):
        #stock minutiae
        minutiae_ending = []
        minutiae_bifurcation = []

        rows, cols = squelette.shape #recupere nbr lignes et colonnes de l’img

        for i in range(1, rows - 1):

            for j in range(1, cols - 1): #parcourt chaque pixels  !!!sauf bords !!! (car regarde voisins autour) ->sinon bugs

                if squelette[i][j] == 1: #1=blanc

                    '''
                    récupère les 8 voisins 
                    calcule combien sont =1
                    [i][j] ne compte pas
                    '''
                    neighbors = squelette[i-1:i+2, j-1:j+2].flatten()
                    count = np.sum(neighbors) - 1  # suppr [i][j]

                    if count == 1: #1 voisin blanc = ending
                        minutiae_ending.append((j, i))  # x, y

                    elif count == 3:#3 voisin blanc ->biffurcassion
                        minutiae_bifurcation.append((j, i))

        return minutiae_ending, minutiae_bifurcation


    # import+ passage binaire
    squelette = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    _, binaire = cv2.threshold(squelette, 127, 1, cv2.THRESH_BINARY_INV) #passse en binaire pour traitement


    # minutiae
    endings, bifurcations = find(binaire)
    


    # affichage img
    color_image = cv2.cvtColor(squelette, cv2.COLOR_GRAY2BGR) #convertit img en couleur pour dessiner les rondss


    for x, y in endings:
        cv2.circle(color_image, (x, y), 3, (0, 0, 255), 1)  # rouge = ending
    for x, y in bifurcations:
        cv2.circle(color_image, (x, y), 3, (0, 255, 0), 1)  # vert )= bifurcations

    cv2.imwrite(output_filename, color_image)

    plt.figure(figsize=(10, 10))
    plt.imshow(color_image[..., ::-1]) # inversant les canaux BGR → RGB pour matplotlib ??
    plt.title('minutiae detecteees')
    plt.axis('off')
    plt.show()
    return [endings,bifurcations]



