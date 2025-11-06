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
    
    (rouge,vert,bleu) = im.getpixel((xb,yb))
    if((rouge, vert, bleu)==(255,255,255)): #vérifie que ce pixel est bien dans la partie blanche du masque 
       
        xr = xb 
        while((rouge, vert, bleu)==(255,255,255) and (xr <= W)):
            xr = xr + 1 
            (rouge,vert,bleu) = im.getpixel((xr,yb))
        
        xl = xb 
        while((rouge, vert, bleu)==(255,255,255) and (xl >= 0)):
            xl = xl - 1 
            (rouge,vert,bleu) = im.getpixel((xl,yb))
            
        
        yu = yb
        while((rouge, vert, bleu)==(255,255,255) and (yu>=0)):
            yu = yu - 1 
            (rouge,vert,bleu) = im.getpixel((xb,yu))
            
        yl = yb
        while((rouge, vert, bleu)==(255,255,255) and (yl<=H)):
            yl = yl + 1 
            (rouge,vert,bleu) = im.getpixel((xb,yl))
    
        xf = (xr + xl) / 2
        yf = (yu + yl) / 2 


    # realignment direction  
    
    im = Image.open(mask_image)
    W , H = im.size
    
    xu = xf 
    (rouge,vert,bleu) = im.getpixel((xu,yu))
    while((rouge, vert, bleu)==(255,255,255) and (xu <= W)):
            xu = xu + 1 
            (rouge,vert,bleu) = im.getpixel((xu,yu))
    
    C1 = xu - xf 
    xu = xf 
    
    while((rouge, vert, bleu)==(255,255,255) and (xu<=0)):
            xu = xu + 1
            (rouge,vert,bleu) = im.getpixel((xu,yu))
            
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
    if( not(D = 0)):
        C4 = yf - yu 
        if(D = 1):
            C3 = C2 - C1
        elif(D = 2): 
            C3 = C2 + C1 
        Q = np.arctan((C2 + C3)/C4)
    
    return Q 