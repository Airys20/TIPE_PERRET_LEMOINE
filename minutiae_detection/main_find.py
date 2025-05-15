
from pretraitement import pretraitements
from find_minutiae import find_minuatiae

filename = 'minutiae_detection\\input\\empreinte4.jpg'
output_filename = 'minutiae_detection\\output\\empreinte4_minutiae.jpg' #[ ] a obtenir automatiquement avec str_modify a l'interieur de find_minutiae


pretraitée_file_name = pretraitements(filename)
find_minuatiae(pretraitée_file_name,output_filename)




