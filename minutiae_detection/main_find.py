
from pretraitement import pretraitements
from find_minutiae import find_minuatiae
import sys
import os

# methode pour acceder a utile ??
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from json_utils import ajouter_personne

filename = 'minutiae_detection\\input\\empreinte4.jpg'
output_filename = 'minutiae_detection\\output\\empreinte4_minutiae.jpg' #[ ] a obtenir automatiquement avec str_modify a l'interieur de find_minutiae


pretraitée_file_name = pretraitements(filename)
tab = find_minuatiae(pretraitée_file_name,output_filename) 
 #[]faire en sorte que ce soit [ [endings [(x,y),orientation] ] , [ biffurcation [(x,y),orientation] ]  ]

print(tab)

ajouter_personne("personne_test",tab)



