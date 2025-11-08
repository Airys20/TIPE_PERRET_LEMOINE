"""
. memo
code permet de determiner l'orientation des zone de l'img = block de taille W_BLOCK  
 etape 1: pretrzitement simple (=normalize + grayscale)
 etape 2: masque = masque autour de l'empreinte
 etape 3: orientation 
"""
"""
.documentation CV2

THRESH_OTSU : treshold où la valeur seuil est determiner automatiquement et n'est donc pas arbitraire
TRESH_BINARY : treshold binaire si pix sup a la val seuil alors passe a val max sinon =0
               ⚠️prend que des img src en niv de gris 
getStructuringElement(cv2.MORPH_ELLIPSE, (15,15)) : creer un "element" de la forme puis taille demandée 

"""             

img = "minutiae_detection\input\empreinteS3_rota25.jpg" 


import numpy  as np
import cv2
from pathlib   import Path
from PIL import Image 


#. Variable reglables 
FICHIER_OUT = "minutiae_detection\\output_orientation"
W_BLOCK =16    
LOW_PASS_FILTER_SIZE=5  #taille lissageetape 4 de l'orientation
COEF_FLOU= 1.0   #écart-type du flou gaussien avant Sobel    
USE_TANGENT=False 


#. Pretraitement de base


def niv_de_gris(path):

    img_nivgris = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img_nivgris  is not None: 
        return img_nivgris

"""
Le but de normilise est d'imposer une moyenne et variance a atteindre ici M0 et VAR0 

formule que l'on veut traduire  (cf article)
$$
\
G(i,j) =
\begin{cases}
M_0 + \sqrt{\dfrac{VAR_0 \, (I(i,j) - M)^2}{VAR}}, & \text{si } I(i,j) > M \\[1.2em]
M_0 - \sqrt{\dfrac{VAR_0 \, (I(i,j) - M)^2}{VAR}}, & \text{sinon.}
\end{cases}
\

\
M = \dfrac{1}{N} \sum_{i,j} I(i,j),
\qquad
VAR = \dfrac{1}{N} \sum_{i,j} \bigl(I(i,j) - M\bigr)^2
\
$$
"""



def normalise_fun(img_grise, M0=100.0, VAR0=100.0):

    I = img_grise.astype(np.float64) #passe l'image en TABLEAU de la val de chaque pixel 
    
    if I.max() <= 1.0: 
        I *= 255.0 #on elargit les niv de gris 

    M = I.mean(); #val moy de gris
    VAR = I.var(); #variance moy de gris 

    if VAR < 1e-9: #cas si variance null on rempli tt pour eviter la div par 0 
        G = np.full_like(I, fill_value=M0, dtype=np.float64)  
    else:
        

        d = I - M
        ajustement = np.sqrt((VAR0 * (d**2)) / VAR) #=TABLEAU des parties sous la racine pour chaque pixel
        #(on conserve le signe p/r a la moyenne mais si + que M on le rend + que M0 et inv)
        G = np.where(I>M, M0+ ajustement, M0 - ajustement ) #CREER un nouv TABLEAU et rempli selon condition : np.where(condition, si sup a la moyenne, si inf a la moy)
    return np.clip(G, 0, 255).astype(np.uint8) #recadre entre [0,255 ] et repasse format uint8 ⚠️sinon bug



#. isolement empreinte (pas dans article mais bug sur orientation sinon)

"""
on veut crree un masque binaire pour isoler empreintre :
0= masque
255= empreinte
"""
def masque_fun(img_grise):
    flou = cv2.GaussianBlur(img_grise, (0,0), 3.0) 
    _, mask = cv2.threshold(flou, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU) #treshold avec valeur optimal deduite par l'algo renvoi val , masque 
    #on veut le masque en blanc ?
    if np.sum(mask==255) > np.sum(mask==0): #si + de noir que de blanc
        mask = 255 - mask #on inv
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15,15))  #COMMENT :⚠️si trop aggressif baissé ou augmenter la taille (memo 15 ok la plupart du temps)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k) #enleve petits trous
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k) #supprime petits points


    #PARTIE TROUVER DANS UN ARTICLE, marche mais jsp comment 
    num,lbl,stats,_ = cv2.connectedComponentsWithStats((mask>0).astype(np.uint8), 8)
    if num > 1:
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask = (lbl==largest).astype(np.uint8)*255
    


    mask = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)), 1) #lisse le tour
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





#. code principal


gray = niv_de_gris(img)
tab_normal = normalise_fun(gray)
cv2.imwrite(FICHIER_OUT + "\\normalized.png", tab_normal)

masque = masque_fun(tab_normal)
cv2.imwrite(FICHIER_OUT+"\\masque.png",masque)
O_bloque=fun_orientation(tab_normal, masque=masque,w=W_BLOCK,low_pass_size=LOW_PASS_FILTER_SIZE, coef_flou=COEF_FLOU)
empreinte=affichage_orient(tab_normal, O_bloque,masque,w=W_BLOCK)




cv2.imwrite(FICHIER_OUT+"\\orientation_empreinte.png", empreinte)
np.save(FICHIER_OUT+"\\O_bloque.npy", O_bloque) #a voir comment reutiiser pour associer orientation <=> minutiae d
print("OK")

