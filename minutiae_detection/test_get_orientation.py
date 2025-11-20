import numpy    as np
X = np.load('minutiae_detection\\output_orientation\\O_bloque.npy', mmap_mode='r')
for i in X:
   
        for j in i:
            if j>0:
                print (j)