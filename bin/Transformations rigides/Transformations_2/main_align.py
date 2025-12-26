import masque
import Re_alignment as ra 
import cv2

def main(img_path): 
    masque_path = masque.main(img_path)
    Q = ra.TFCP(masque_path)
    print(Q)
    img=cv2.imread(img_path)
    hauteur = img.shape[0] 
    largeur = img.shape[1]
    angle=Q
    centre = (largeur / 2, hauteur / 2)
    M = cv2.getRotationMatrix2D(centre, angle, 1.0)
    nouvelle_image=cv2.warpAffine(img, M, (largeur,hauteur))
    cv2.imshow('image',nouvelle_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    
    
main('Global Matching\\Tests\\syria_rotate.jpg')