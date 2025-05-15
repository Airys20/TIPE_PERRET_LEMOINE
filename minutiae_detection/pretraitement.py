import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize 
from scipy.ndimage import binary_opening #supprime bruit
                                         #lisse les bords
                                         #ne touche pas ce qui esr bien formé

import os # permet d'utiliser fichiers et répertoires et de manipuler les chemins de fichiers



#nom du fichier entree+ sortie (voir si on peut automatiser)
filename = 'minutiae_detection\input\empreinte4.jpg' test
output_filename = 'minutiae_detection\pretraitees\empreinte4_pretraitee.jpg'

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
filtered = cv2.bilateralFilter(contrast, 9, 100, 100) #lisse img mais garde contours
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
cleaned = binary_opening(morph_bin, structure=np.ones((3, 3))) #comme morph_open (l54) mais plus math -> mieux pour squeletisation

#squelettisation
#Each connected component in the image is reduced to a single-pixel wide skeleton
skeleton = skeletonize(cleaned) #-> ensemble de 0 et 1 donc par visualisable ?

# conversion en image enregistrable 
skeleton_img = (skeleton * 255).astype(np.uint8) #transforme les 0 et 1 en vrai picel couleur en multipliant par 255
inverted = cv2.bitwise_not(skeleton_img) #inverse couleur pour faciliter traitement futur 

# sauvegadre
cv2.imwrite(output_filename, inverted)
print("Image prétraitée enregistrée ")

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
