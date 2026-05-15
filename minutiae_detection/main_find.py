import numpy  as np
import sys
import os
import cv2
import math

from pretraitement import pretraitements 
from find_minutiae2 import find_minuatiae 
from orientation import main_orientation 
# methode pour acceder a utile ??
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from json_utils import ajouter_personne



#REGLAGES 
NOM = 'Marcel'
BASE = "recherche.json"
#FILENAME='minutiae_detection\input\empreinte_feutre.jpeg '#lien ou tu mets l img a tester
FILENAME='minutiae_detection\\input\\Marcel2.png'#lien ou tu mets l img a tester
output_filename = 'minutiae_detection/passage_main/' + NOM + '_output_minutiae.jpg' #lien ou tu stocke l'img superposée des minuties



#################### SYRIA PAS TOUCHE :) ###############

def ajouter_orientation_aux_minuties(minuties, O_bloque, H_orig, W_orig, w_block=16):
    """
    minuties: [[x_n,y_n], typ, [dx,dy]] sur [0;1]
    O_bloque:blocs sur img (H_orig x W_orig)
    """

    Hb, Wb = O_bloque.shape
    out = []
    for (x_n, y_n), typ, (dx, dy) in minuties:
        
        x = int(round(x_n *(W_orig - 1)))# on repasse en px
        y = int(round(y_n*(H_orig - 1)))

       #on trouve les numeros de bloc de l'orientation correspondante
        bi = y // w_block
        bj = x // w_block

       
        bi = max(0, min(Hb - 1, bi)) #doubkleee chexk
        bj = max(0, min(Wb - 1, bj))

        theta = float(O_bloque[bi, bj])
        out.append([[x_n, y_n], typ, [dx, dy], theta])

    return out






img0 = cv2.imread(FILENAME, cv2.IMREAD_GRAYSCALE)
H_orig, W_orig = img0.shape #recup mesure originale AVANT resize)

pretraitée_file_name = pretraitements(FILENAME)
main_orientation(FILENAME)
O_bloque = np.load('minutiae_detection/output_orientation/O_bloque.npy')

file_masque= 'minutiae_detection/output_orientation/masque.png'
tab = find_minuatiae(pretraitée_file_name,output_filename, mask_filename=file_masque) 

tab = ajouter_orientation_aux_minuties(tab, O_bloque, H_orig, W_orig) 

ajouter_personne(NOM,tab,BASE)


#np.set_printoptions(threshold=sys.maxsize)
#print(O_bloque)


