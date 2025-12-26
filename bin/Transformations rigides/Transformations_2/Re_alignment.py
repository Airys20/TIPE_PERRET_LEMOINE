import numpy as np
from PIL import Image

'''
$$
x_b = 0.5 \times W \space \space y_b = 0.5 \times H 
$$
$$
x_f = \frac{(x_r + x_f)}{2} \space \space y_f = \frac{(y_u + y_t)}{2} 
$$

'''
def TFCP(mask_image):
    
    im = Image.open(mask_image)
    
    W , H = im.size
    
    xb = 0.5 * W
    yb = 0.5 * H 
    
    p = im.getpixel((xb,yb))
    print(p)
    
    if p == 255: #vérifie que ce pixel est bien dans la partie blanche du masque 
       
        xr = xb 
        while(p == 255 and (xr <= W)):
            xr = xr + 1 
            p = im.getpixel((xr,yb))
        
        xl = xb 
        p = im.getpixel((xb,yb))
        while(p == 255 and (xl >= 0)):
            xl = xl - 1 
            p = im.getpixel((xl,yb))
            
        
        yu = yb
        p = im.getpixel((xb,yb))
        while( p == 255 and (yu>=0)):
            yu = yu - 1 
            p = im.getpixel((xb,yu))
            
        yl = yb
        p = im.getpixel((xb,yb))
        while( p == 255 and (yl<=H)):
            yl = yl + 1 
            p = im.getpixel((xb,yl))
    
        xf = (xr + xl) / 2
        yf = (yu + yl) / 2 
    else : 
        print("Erreur, point central de l'image pas dans le blanc")
        
    print(yu,yl, yf)

    # realignment direction  
    
    im = Image.open(mask_image)
    W , H = im.size
    
    xu = xf 
    p = im.getpixel((xu,yu))
    while(p == 255 and (xu <= W)):
            xu = xu + 1 
            p = im.getpixel((xu,yu))
    
    C1 = xu - xf 
    xu = xf 
    
    while(p == 255 and (xu<=0)):
            xu = xu + 1
            p = im.getpixel((xu,yu))
            
    C2 = xf - xu 
    
    if(C1<C2):
        D = 1 #clockwise 
    elif(C2<C1):
        D = 2 #anti-clockwise
    else :
        D = 0 #D is upright 
    
    xu = xf 
    
    # angle
    
    Q = 0 
    if( D != 0):
        C4 = yf - yu 
        if( D == 1):
            C3 = C2 - C1
        elif(D == 2): 
            C3 = C2 + C1 
        Q = np.arctan((C2 + C3)/C4)
    
    return Q 




