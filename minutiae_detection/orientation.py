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


#. isolement empreinte (pas dans article mais bug sur orientation sinon)

"""
on veut crree un masque binaire pour isoler empreintre :
0= masque
255= empreinte
"""


def fill_holes_safe(imgbin):
    """
    Remplit les trous d'un masque 0/255 de manière robuste.
    Astuce : on force une bordure noire pour garantir que le fond est connecté au bord.
    """
    m = imgbin.copy()
    h, w = m.shape

    # force une bordure noire (très important)
    m[0, :] = 0
    m[-1, :] = 0
    m[:, 0] = 0
    m[:, -1] = 0

    flood = m.copy()
    mask_ff = np.zeros((h + 2, w + 2), np.uint8)

    # floodfill depuis (0,0) qui est maintenant garanti fond (0)
    cv2.floodFill(flood, mask_ff, (0, 0), 255)

    # trous = zones non atteintes par floodfill
    holes = cv2.bitwise_not(flood)
    filled = cv2.bitwise_or(m, holes)
    return filled


def masque_fun_v3(img_grise, debug=False):
    """
    Masque ROI doigt robuste :
    0 = fond
    255 = empreinte
    """
    # 1) flou fort pour effacer les crêtes (forme globale)
    flou_gros = cv2.GaussianBlur(img_grise, (0, 0), 25.0)

    # 2) Otsu
    thr, mask = cv2.threshold(flou_gros, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 3) décider l’inversion en regardant les coins (le fond est aux coins)
    corners = np.array([mask[0,0], mask[0,-1], mask[-1,0], mask[-1,-1]])
    # si la majorité des coins est blanche, alors le fond est blanc -> on inverse
    if np.mean(corners) > 127:
        mask = 255 - mask

    if debug:
        print("Otsu thr =", thr, "| white ratio after otsu =", np.mean(mask==255))

    # 4) fermeture morpho (colle les zones, bouche trous)
    kclose = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kclose, iterations=2)

    if debug:
        print("white ratio after close =", np.mean(mask==255))

    # 5) plus grande composante connexe
    num, lbl, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), 8)
    if num > 1:
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask = (lbl == largest).astype(np.uint8) * 255

    if debug:
        print("white ratio after CC =", np.mean(mask==255))

    # 6) remplissage trous (safe)
    mask = fill_holes_safe(mask)

    if debug:
        print("white ratio after fill holes =", np.mean(mask==255))

    # 7) dilatation légère (optionnelle) pour éviter masque trop “serré”
    kdil = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask = cv2.dilate(mask, kdil, iterations=1)

    if debug:
        print("white ratio after dilate =", np.mean(mask==255))

    # 8) lissage bords léger
    kopen = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kopen, iterations=1)

    if debug:
        print("white ratio final =", np.mean(mask==255))

    return mask

#. Orientation (Hong) — SOMMES UNIQUEMENT DANS LE MASQUE
# θ = 0.5 * atan2(phix, Vy)  (ordre corrigé) 

def fun_orientation(img_grise, masque, w=8, low_pass_size=5, coef_flou=1.0):
    """
        imggrise + masque -> matrice des orientation par bloc + 
    """

    G = img_grise.astype(np.float32) #repasse l'img en TAB de val
    H= len(G)
    W = len (G[0])
    Hb, Wb = H//w, W//w #etape 1 de l'algo : decoupage en blocs 



    # etape2:gradients 
    Gs = cv2.GaussianBlur(G, (0,0), coef_flou) # permet de lisser un peu ava,t les gradient 

    Gx=cv2.Sobel(Gs,cv2.CV_32F, 1, 0,ksize=3) # on utilise sobel comme demandé dans l'article 
    Gy=cv2.Sobel(Gs, cv2.CV_32F,0, 1,ksize=3) # doc : computes an approximation of the gradient of the image intensity function
    # poids: 0 hors masque, |VI| dans le ROI (stabilise)
    mag = cv2.magnitude(Gx, Gy) #les pixek les plus fonçé seront + imp
    POIDS = (masque > 0).astype(np.float32) * (mag + 0.00001) #COMMENT : ⚠️ attention ne pas oublié le +0,0001 sinon rique div par 0 apres
   
    
    
    
    # CALCULE DES "local orientation of each block centered at pixel i j "
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
            gx = Gx[y0:y1, x0:x1] 
            gy=Gy[y0:y1, x0:x1]
            # Sommes pond. QUE sur masque
            Sxx = np.sum(bloc_poids * (gx*gx))
            Syy = np.sum(bloc_poids * (gy*gy))
            Sxy =np.sum(bloc_poids *(gx*gy))
            # Vx = 2 Σ Gx Gy ; Vy = Σ(Gx^2 - Gy^2)
            Vx[bi,bj] = 2.0 * Sxy
            Vy[bi,bj] = (Sxx - Syy)




    # CALCULE DE THETA 

    """$$
    \
    \theta = \tfrac{1}{2} \, \tan^{-1}\!\left( \frac{V_x}{V_y} \right)
    \
    $$
    """
    theta = 0.5 * np.arctan2(Vx, Vy) #on utilise 2 pour avoir qqch appartenant a [-180,180] (sino [-90,90])

    # etape 4  : LOW PASS FILTER
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
    phix = np.cos(2*theta)
    phiy = np.sin(2*theta)


    if low_pass_size > 1: # phi' juste remplacer par des flou parsque jsp quoi faire sinon ;-;



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
    empreinte = cv2.cvtColor(img_grise, cv2.COLOR_GRAY2BGR) #=img de fo nd
    taille_trait =int (0.4*w) 

    for bi in range(0, Hb): #on regarde tt les centre de blocs 
        for bj in range(0, Wb):
            centre_y=int(bi*w+w/2)
            centre_x= int(bj*w+w/2)

            if (masque[centre_y,centre_x]<=0): #on verifie que appartient bien a l'emprie=nte pour pas dessine autour
                continue




            
            th = (float(orient_bloc[bi,bj])+np.pi/2.0) % np.pi #sinon fzit l'opposé mais jsp pourquoi ;-;

            #CALCUL EXTREMITES SEGMENTS
            dy=int(taille_trait*np.sin(th))
            dx =int(taille_trait*np.cos(th))


            y1,x1=centre_y-dy,centre_x-dx #coordonée e seglent  
            y2,x2=centre_y+dy,centre_x+dx


            if 0<=y1<H and 0<=y2<H and 0  <=x1<W and 0 <=x2<  W: #condition de tracage => que sdi ds l'eimg

                cv2.line(empreinte,(x1,y1),(x2,y2),(0,255,0),1) 
                #COMMENT : trester cv2.LINE_AA une fois que ça marche

    return empreinte


#. Orientation matching (non utilisé pour l'instant)
def orientation_matching(w, coord ):
    x,y = coord
    O_bloque = np.load('minutiae_detection\\output_orientation\\O_bloque.npy', mmap_mode='r')
    bi = y// w
    bj = x// w
    return O_bloque[bi, bj]



'''
#. Ridge frequency image 

def ridge_freq_fun(img_grise,masque,w, O_bloque):
    G = img_grise.astype(np.float32) #repasse l'img en TAB de val
    H= len(G)
    W = len (G[0])
    Hb, Wb = H//w, W//w
    X = np.zeros(w)
    for bi in range(0, Hb): #on regarde tt les centre de blocs 
        for bj in range(0, Wb):
            centre_y=int(bi*w+w/2)
            centre_x= int(bj*w+w/2)
            for d in range (0,w-1):
                u= centre_x  + (d-w/2)*np.cos( O_bloque[centre_x][centre_y]) + ()
                X[bi*w+bj] = 1/w * 
 
           
'''



    


#. code principal

def main_orientation(img):
    gray = niv_de_gris(img)
    tab_normal = normalise_fun(gray)
    cv2.imwrite(FICHIER_OUT + "\\normalized.png", tab_normal)

    masque = masque_fun_v3(tab_normal)
    cv2.imwrite(FICHIER_OUT+"\\masque.png",masque)
    O_bloque=fun_orientation(tab_normal, masque=masque,w=W_BLOCK,low_pass_size=LOW_PASS_FILTER_SIZE, coef_flou=COEF_FLOU)
    empreinte=affichage_orient(tab_normal, O_bloque,masque,w=W_BLOCK)


    

    cv2.imwrite(FICHIER_OUT+"\\orientation_empreinte.jpg", empreinte)
    image = Image.open(FICHIER_OUT+"\\orientation_empreinte.jpg")
    image.show()
    np.save(FICHIER_OUT+"\\O_bloque.npy", O_bloque) #a voir comment reutiiser pour associer orientation <=> minutiae d
    print("ORIENTATION OK")
    print(orientation_matching(W_BLOCK, (200,150) ))
