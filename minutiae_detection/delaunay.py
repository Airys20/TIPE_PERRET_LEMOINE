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


filename = 'minutiae_detection\\input\\empreinteS3.jpg'
output_filename = 'minutiae_detection\\output\\empreinteS3_minutiae.jpg' 


pretraitée_file_name = pretraitements(filename)

tab = find_minuatiae(pretraitée_file_name,output_filename) 

points = np.zeros(shape=(len(tab), 2))
for i in range (len(tab)) :
    
    points[i] = tab[i][0]
    #np.append(points , [minutiae[0]])


#ENLEVE POINTS ABSURDE
# [ ] a decaler au traitement des minutiae 
center = np.mean(points, axis=0)
distances = np.linalg.norm(points - center, axis=1)
threshold = np.mean(distances) + 2 * np.std(distances)
mask = distances < threshold
filtered_points = points[mask]

print(filtered_points)


tri = Delaunay(filtered_points)

plt.figure(figsize=(10, 10))
plt.triplot(filtered_points[:,0], filtered_points[:,1], tri.simplices, linewidth=1)
plt.plot(filtered_points[:,0], filtered_points[:,1], 'ro', markersize=1)
plt.show()