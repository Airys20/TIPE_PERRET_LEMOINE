import numpy  as np
import sys
import os
import cv2
import math

from pretraitement import pretraitements
from find_minutiae import find_minuatiae
from orientation import main_orientation 
# methode pour acceder a utile ??
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from json_utils import ajouter_personne



#REGLAGES 
NOM = 'Syria'
BASE = "catalogue.json"
FILENAME='minutiae_detection\input\\Syria_catalogue.jpeg' #lien ou tu mets l'img a tester
output_filename = 'minutiae_detection/passage_main/' + NOM + '_output_minutiae.jpg' #lien ou tu stocke l'img superposée des minuties



#################### SYRIA PAS TOUCHE :) ###############

def ajouter_orientation_aux_minuties_repere_original(minuties, O_bloque, H_orig, W_orig, w_block=16):
    """
    minuties: [[x_n,y_n], typ, [dx,dy]] où x_n,y_n sont normalisés (0..1) sur l'image 512x512
    O_bloque: orientation par blocs calculée sur l'image originale (H_orig x W_orig)
    Ajoute theta = O_bloque[bi,bj] (radians)
    """
    Hb, Wb = O_bloque.shape
    out = []

    for (x_n, y_n), typ, (dx, dy) in minuties:
        # 1) normalisé -> pixels de l'image ORIGINALE
        x = int(round(x_n * (W_orig - 1)))
        y = int(round(y_n * (H_orig - 1)))

        # 2) pixels -> bloc
        bi = y // w_block
        bj = x // w_block

        # 3) clamp sécurité
        bi = max(0, min(Hb - 1, bi))
        bj = max(0, min(Wb - 1, bj))

        theta = float(O_bloque[bi, bj])
        out.append([[x_n, y_n], typ, [dx, dy], theta])

    return out






img0 = cv2.imread(FILENAME, cv2.IMREAD_GRAYSCALE)
H_orig, W_orig = img0.shape


pretraitée_file_name = pretraitements(FILENAME)
main_orientation(FILENAME)
O_bloque = np.load('minutiae_detection/output_orientation/O_bloque.npy')

file_masque= 'minutiae_detection/output_orientation/masque.png'
tab = find_minuatiae(pretraitée_file_name,output_filename, mask_filename=file_masque) 

tab = ajouter_orientation_aux_minuties_repere_original(tab, O_bloque, H_orig, W_orig) 

ajouter_personne(NOM,tab,BASE)
np.set_printoptions(threshold=sys.maxsize)
#print(O_bloque)


