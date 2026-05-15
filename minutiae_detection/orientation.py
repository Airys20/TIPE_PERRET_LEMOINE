"""
. memo
permet de determiner orientation des zone de l'img 
=> block de taille W_BLOCK  
 etape 1: pretraitement simple (=normalize + grayscale)
 etape 2: masque = masque autour de l'empreinte
 etape 3: orientation 
"""
"""
.documentation CV2

THRESH_OTSU : treshold avec seuil automatique
TRESH_BINARY : treshold binaire 
"""             

img = "C:/Users/Elise/Downloads/empreinte_overlined.jpeg" 

from PIL import Image
import numpy  as np
import cv2
from pathlib   import Path
from PIL import Image 
from masque import masque_fun_v3



#. Variable reglables 
FICHIER_OUT = "minutiae_detection\\output_orientation"
W_BLOCK =16    
LOW_PASS_FILTER_SIZE=5  #taille lissage  etape 4 de l'orientation
COEF_FLOU= 1.0   #écart-type du flou gaussien avant Sobel    
USE_TANGENT=False 


#. Pretraitement de base


def niv_de_gris(path):

    img_nivgris = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    #img_resize = cv2.resize(img_nivgris, (512, 512))  # resize pr meilleur "generalisation" <- pas forcement le bon mot
    if img_nivgris  is not None: 
        return img_nivgris


def normalise_fun(img_grise, M0=100.0, VAR0=100.0):

    #passe l'image en TABLEAU de la val de chaque pixel 
    I = img_grise.astype(np.float64) 
    
    if I.max() <= 1.0: 
        I *= 255.0 #on elargit les niv de gris 

    M = I.mean(); #val moy de gris
    VAR = I.var(); #variance moy de gris 

    if VAR < 1e-9: #eviter div par 0 
        G = np.full_like(I, fill_value=M0, dtype=np.float64)  
    else:
        d = I - M
        ajustement = np.sqrt((VAR0 * (d**2)) / VAR) 
        #pour chaque pixel : écart normalisé 
        #conserve l'amplitude relative p/r à la moyenne, 
        # mais à échelle de la variance cible VAR0

        G = np.where(I>M, M0+ ajustement, M0 - ajustement ) 
        # construit img normalisée pixel par pixel :
        #   plus clair que M  → M0 + ajustement 
        #   plus sombre que M → M0 - ajustement 
        #-> signe conservé, amplitude remise vers VAR0
    return np.clip(G, 0, 255).astype(np.uint8) 
    #recadre entre [0,255 ] et passe format uint8 sinon bug



#. Orientation (Hong)  

def fun_orientation(img_grise, masque, w=8, low_pass_size=5, coef_flou=1.0):
    """
        img grise + masque -> matrice des orientation par bloc + 
    """

    G = img_grise.astype(np.float32) #repasse l'img en TAB de val
    H= len(G)
    W = len (G[0])
    Hb, Wb = H//w, W//w #etape 1 de l'algo : decoupage en blocs 


    # etape2:gradients 

    # lisser un peu avant gradients 
    Gs = cv2.GaussianBlur(G, (0,0), coef_flou) 

    #gradient horizontaux et verticaux
    Gx=cv2.Sobel(Gs,cv2.CV_32F, 1, 0,ksize=3) 
    Gy=cv2.Sobel(Gs, cv2.CV_32F,0, 1,ksize=3) 

    # poids: 0 hors masque, |VI| dans le ROI 
    mag = cv2.magnitude(Gx, Gy) # + fonçé → + imp
    POIDS = (masque > 0).astype(np.float32) * (mag + 0.00001) #rique div par 0 +0.00001
   
    
    
    
    # calcul "local orientation of each block centered at pixel i j "
    Vx =np.zeros((Hb, Wb),np.float32)
    Vy =np.zeros((Hb, Wb),np.float32)

    """
    on utilise ces equations :
    $$
    \
\begin{aligned}
S_{xx} &= \sum_{i,j} w_{i,j}\, G_x(i,j)^2, \\[0.5em]
S_{yy} &= \sum_{i,j} w_{i,j}\, G_y(i,j)^2, \\[0.5em]
S_{xy} &= \sum_{i,j} w_{i,j}\, G_x(i,j)\, G_y(i,j),
\end{aligned}

\
V_x = 2 S_{xy}, \qquad V_y = S_{xx} - S_{yy}.
\
$$
    """ 
    for bi in range(Hb):
        for bj in range(Wb):
            
            #bord du block
            y0,y1 = bi*w,(bi+1)*w
            x0,x1 = bj*w,(bj+1)*w
            
            #poid du bloc
            bloc_poids = np.zeros((y1 - y0, x1 - x0), dtype=POIDS.dtype)

            for i in range(y0, y1):
                for j in range(x0, x1):
                    bloc_poids[i - y0, j - x0] = POIDS[i, j]

            if not np.any(bloc_poids>0): # si aucun poid du bloc >0 alors on saute bloc-> vide pas orieneteation fiable 
                continue

            #gradient du bloc 
            gx = Gx[y0:y1, x0:x1] #Gx = cos(θ + 90°) = −sin(θ)
            gy=Gy[y0:y1, x0:x1] #Gy = sin(θ + 90°) = cos(θ)
            
            #sommes pondérées
            Sxx = np.sum(bloc_poids * (gx*gx)) #Sxx = Σ Gx² = Σ sin²(θ)
            Syy = np.sum(bloc_poids * (gy*gy)) #Syy = Σ Gy² = Σ cos²(θ)
            Sxy =np.sum(bloc_poids *(gx*gy)) 
            #Sxy = Σ Gx·Gy = Σ (−sin(θ))·cos(θ) = −Σ sin(θ)cos(θ)
            
            '''
            Rappels :
               sin(θ)·cos(θ) = ½·sin(2θ)
               sin²(θ) = ½·(1 - cos(2θ))
               cos²(θ) = ½·(1 + cos(2θ))

            => on veut 2θ donc :
             Sxx - Syy = N·½(1-cos2θ) - N·½(1+cos2θ) = -N·cos(2θ)
             2·Sxy = 2·(-N·½·sin(2θ)) = -N·sin(2θ)
             (au signe et facteur N près on a)
             Vy = Sxx-Syy encode cos(2θ) 
             Vx = 2·Sxy encode sin(2θ) 
            '''
            Vx[bi,bj] = 2.0 * Sxy
            Vy[bi,bj] = (Sxx - Syy)




    # CALCULE DE THETA 

    #$$\theta = \tfrac{1}{2} \, \tan^{-1}\!\left( \frac{V_x}{V_y} \right) $$
    
    theta = 0.5 * np.arctan2(Vx, Vy) #appartenant a [-180,180] 

    
    # etape 4  : FILTRE PASSE - BAS 
    #lisser mais en faisant attention que 179° et 1° sont les meme pentes 
    #on repasse ne 2theta pour "deplier"
    
    """
    $$
    \
    \begin{aligned}
    \Phi_x(i,j) &= \cos\big(2\,\theta(i,j)\big), \quad \text{and} \\[0.5em]
    \Phi_y(i,j) &= \sin\big(2\,\theta(i,j)\big),
    \end{aligned}
    \
    \
    \begin{aligned}
    \Phi_x'(i,j) &= 
    \sum_{u=-w_\Phi/2}^{w_\Phi/2}
    \sum_{v=-w_\Phi/2}^{w_\Phi/2}
    W(u,v)\,\Phi_x(i - u w,\, j - v w), \quad \text{and} \\[0.8em]
    \Phi_y'(i,j) &= 
    \sum_{u=-w_\Phi/2}^{w_\Phi/2}
    \sum_{v=-w_\Phi/2}^{w_\Phi/2}
    W(u,v)\,\Phi_y(i - u w,\, j - v w),
    \end{aligned}
    \
    $$    
    """
    phix = np.cos(2*theta) #val dans [-1,1] pour chaque bloc
    phiy = np.sin(2*theta)


    if low_pass_size > 1: # phi' juste remplacer par des flou 

        #blur = somme pondéré des voisins de phi = phi'
        #passe bas = attenue les variations rapides de phi = angles abérants
        phix=cv2.blur(phix,(low_pass_size, low_pass_size)) 
        phiy=cv2.blur(phiy, (low_pass_size,low_pass_size))  
    
    
    O_bloc = 0.5*np.arctan2(phiy, phix) #=tab de l'orientation par bloc
    
    
    return O_bloc


#.affichage orientation

def affichage_orient(img_grise, orient_bloc, masque, w=8):
    
    H = len(img_grise)
    W = len(img_grise[0])
    Hb= len(orient_bloc)
    Wb =len (orient_bloc[0])

    empreinte = cv2.cvtColor(img_grise, cv2.COLOR_GRAY2BGR) #=img de fond
    taille_trait =int (0.4*w) 

    for bi in range(0, Hb): #on regarde tout les centre de blocs 
        for bj in range(0, Wb):

            centre_y=int(bi*w+w/2)
            centre_x= int(bj*w+w/2)
            
            if (masque[centre_y,centre_x]<=0): #on verifie que appartient bien a l'emprie=nte pour pas dessine autour
                continue
            
            th = (float(orient_bloc[bi,bj])+np.pi/2.0) % np.pi #axe y de open cv inversé donc ajoute pi/2 

            #CALCUL EXTREMITES SEGMENTS
            dy=int(taille_trait*np.sin(th))
            dx =int(taille_trait*np.cos(th))


            y1,x1=centre_y-dy,centre_x-dx #coordonée segment  
            y2,x2=centre_y+dy,centre_x+dx


            if 0<=y1<H and 0<=y2<H and 0  <=x1<W and 0 <=x2<  W: #condition de tracage => que si dans l'img

                cv2.line(empreinte,(x1,y1),(x2,y2),(0,255,0),1) 
                

    return empreinte



#. code principal

def main_orientation(img):

    gray = niv_de_gris(img)
    tab_normal = normalise_fun(gray)
    cv2.imwrite(FICHIER_OUT + "\\normalized.png", tab_normal)

    masque = masque_fun_v3(tab_normal)
    cv2.imwrite(FICHIER_OUT+"\\masque.png",masque)
    O_bloque=fun_orientation(tab_normal, masque=masque,w=W_BLOCK,\
                             low_pass_size=LOW_PASS_FILTER_SIZE, coef_flou=COEF_FLOU)
    
    empreinte=affichage_orient(tab_normal, O_bloque,masque,w=W_BLOCK)

    cv2.imwrite(FICHIER_OUT+"\\orientation_empreinte.jpg", empreinte)
    image = Image.open(FICHIER_OUT+"\\orientation_empreinte.jpg")
    image.show()
    #sauvegarde de la matrice d'orientation par bloc 
    np.save(FICHIER_OUT+"\\O_bloque.npy", O_bloque) 
    print("ORIENTATION OK")

