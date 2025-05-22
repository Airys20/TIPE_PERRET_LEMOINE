"""
    [x] recuperer tab entree et catalogue 
    [x] triangulation de delaunay 
    [x] afficher delaunay
    [ ] recup tab des donnees des edges 
"""
from pretraitement import pretraitements
from find_minutiae import find_minuatiae
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import Delaunay

filename = 'minutiae_detection\\input\\empreinte1.jpg'
output_filename = 'minutiae_detection\\output\\empreinte1_minutiae.jpg' 


pretraitée_file_name = pretraitements(filename)
tab = find_minuatiae(pretraitée_file_name,output_filename) 

points = np.zeros(shape=(len(tab), 2))
for i in range (len(tab)) :
    
    points[i] = tab[i][0]
    #np.append(points , [minutiae[0]])

print(points)

tri = Delaunay(points)

plt.figure(figsize=(10, 10))
plt.triplot(points[:,0], points[:,1], tri.simplices, linewidth=1)
plt.plot(points[:,0], points[:,1], 'ro', markersize=1)
plt.show()