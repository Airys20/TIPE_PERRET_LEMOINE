import cv2
import numpy as np
from scipy.ndimage import convolve


# ============================================================
#  ETAPE 1 : Estimation de la fréquence locale
# ============================================================
# 
# Idée (cf. article Hong et al. section 2.5) :
#   Pour chaque bloc (bi, bj) on projette les niveaux de gris
#   PERPENDICULAIREMENT aux crêtes (i.e. dans la direction de
#   l'orientation) sur une "x-signature".
#   Cette x-signature est une onde quasi-sinusoïdale dont
#   la fréquence = espacement inter-crêtes.
#   On détecte les pics de cette signature → T = nbre de pixels
#   entre deux pics consécutifs → f = 1/T
#
# Paramètres de l'article :
#   w_bloc = 16  (taille du bloc d'orientation)
#   l_w    = 32  (longueur de la fenêtre orientée)
#   w_w    = 16  (largeur de la fenêtre orientée)
#   plage valide pour 500 dpi : [1/3 ; 1/25]
#
# ⚠️  Si aucun pic n'est trouvé dans le bloc → on met -1
#      (sera interpolé depuis les voisins ensuite)

def estimer_freq_bloc(G, theta_bloc, bi, bj, w=16, lw=32):
    """
    Estime la fréquence locale pour UN bloc (bi, bj).

    G          : image normalisée float32 (H x W)
    theta_bloc : matrice d'orientation par blocs (Hb x Wb) en radians
    bi, bj     : indices du bloc
    w          : taille du bloc (même que pour l'orientation)
    lw         : longueur de la fenêtre orientée (32 dans l'article)

    Retourne : fréquence f ∈ [1/25, 1/3] ou -1 si non calculable
    """
    H, W = G.shape
    # centre du bloc en pixels
    cy = int(bi * w + w / 2)
    cx = int(bj * w + w / 2)

    theta = float(theta_bloc[bi, bj])

    # vecteur perpendiculaire aux crêtes = direction de l'orientation θ
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    # x-signature : moyenne sur w pixels dans la direction perpendiculaire
    # pour chaque position k le long de l'axe des crêtes
    X = np.zeros(lw)
    valid_cols = 0

    for k in range(lw):
        # décalage le long de l'axe PARALLÈLE aux crêtes (θ + 90°)
        offset_k = k - lw // 2
        col_sum = 0.0
        count = 0

        for d in range(w):
            # décalage le long de l'axe PERPENDICULAIRE (direction θ)
            offset_d = d - w // 2

            # coordonnées du pixel
            # u = direction perpendiculaire aux crêtes (cos θ, sin θ)
            # v = direction parallèle aux crêtes    (-sin θ, cos θ)
            py = int(round(cy + offset_d * sin_t  + offset_k * cos_t))
            px = int(round(cx + offset_d * cos_t  - offset_k * sin_t))

            if 0 <= py < H and 0 <= px < W:
                col_sum += G[py, px]
                count += 1

        if count > 0:
            X[k] = col_sum / count
            valid_cols += 1

    if valid_cols < lw // 2:
        return -1.0  # pas assez de pixels valides

    # détection des pics dans X (maxima locaux)
    pics = []
    for k in range(1, lw - 1):
        if X[k] > X[k - 1] and X[k] > X[k + 1]:
            pics.append(k)

    if len(pics) < 2:
        return -1.0  # pas assez de pics pour mesurer l'espacement

    # T = espacement moyen entre pics consécutifs
    espacements = [pics[i + 1] - pics[i] for i in range(len(pics) - 1)]
    T = np.mean(espacements)

    if T < 1e-3:
        return -1.0

    f = 1.0 / T

    # plage valide pour ~500 dpi d'après l'article [1/3 ; 1/25]
    if f < 1.0 / 25.0 or f > 1.0 / 3.0:
        return -1.0

    return f


def estimer_freq_image(G, theta_bloc, masque, w=16, lw=32):
    """
    Calcule l'image de fréquence bloc par bloc.
    Les blocs invalides (-1) sont interpolés depuis leurs voisins.

    Retourne :
        F_bloc      (Hb x Wb) : fréquence par bloc
        freq_fiable (Hb x Wb) : booléen — True = mesure directe,
                                           False = interpolée (zone douteuse)
    """
    H, W = G.shape
    Hb, Wb = H // w, W // w
    F_bloc = np.full((Hb, Wb), -1.0, dtype=np.float32)

    # --- passe 1 : calcul direct sur les blocs valides (dans le masque) ---
    for bi in range(Hb):
        for bj in range(Wb):
            cy = int(bi * w + w / 2)
            cx = int(bj * w + w / 2)
            if cy >= H or cx >= W or masque[cy, cx] == 0:
                continue
            F_bloc[bi, bj] = estimer_freq_bloc(G, theta_bloc, bi, bj, w, lw)

    # on garde la trace des blocs avec une mesure directe fiable
    # (valeur != -1 après la passe 1, donc pas interpolée)
    freq_fiable = (F_bloc != -1.0)

    # --- passe 2 : interpolation des -1 par moyenne des voisins valides ---
    max_iter = 10
    for _ in range(max_iter):
        invalides = np.argwhere(F_bloc == -1.0)
        if len(invalides) == 0:
            break
        F_new = F_bloc.copy()
        for (bi, bj) in invalides:
            voisins = []
            for dbi in [-1, 0, 1]:
                for dbj in [-1, 0, 1]:
                    ni, nj = bi + dbi, bj + dbj
                    if 0 <= ni < Hb and 0 <= nj < Wb and F_bloc[ni, nj] != -1.0:
                        voisins.append(F_bloc[ni, nj])
            if voisins:
                F_new[bi, bj] = np.mean(voisins)
        F_bloc = F_new

    valides = F_bloc[F_bloc != -1.0]
    f_mediane = float(np.median(valides)) if len(valides) > 0 else 0.1
    F_bloc[F_bloc == -1.0] = f_mediane

    # lissage passe-bas (article : filtre gaussien 7x7)
    F_bloc = cv2.GaussianBlur(F_bloc, (7, 7), 0)

    return F_bloc, freq_fiable


# ============================================================
#  ETAPE 2 : Construction du noyau de Gabor
# ============================================================
#
# Formule de l'article (eq. 18) :
#
#   h(x, y : θ, f) =
#     exp( -1/2 * [ (x·cosθ + y·sinθ)² / σx²
#                 + (-x·sinθ + y·cosθ)² / σy² ] )
#     * cos( 2π·f·(x·cosθ + y·sinθ) )
#
# où :
#   θ  = orientation locale des crêtes (radians)
#   f  = fréquence locale (cycles/pixel)
#   σx = σy = 4.0 (valeur empirique de l'article)
#   taille noyau = 11x11 (article : wg = 11)

def construire_noyau_gabor(theta, f, sigma_x=4.0, sigma_y=4.0, taille=11):
    """
    Construit le noyau 2D du filtre de Gabor pour une orientation et
    fréquence données.

    theta  : orientation en radians
    f      : fréquence en cycles/pixel
    sigma_x, sigma_y : étendues de la gaussienne (article : 4.0)
    taille : taille du noyau (article : 11)

    Retourne un tableau numpy (taille x taille) float32
    """
    demi = taille // 2
    # grille de coordonnées centrée sur 0
    y, x = np.mgrid[-demi:demi + 1, -demi:demi + 1].astype(np.float32)

    # rotation des coordonnées selon l'orientation θ
    # x' = axe perpendiculaire aux crêtes (direction des crêtes)
    # y' = axe parallèle aux crêtes
    x_rot =  x * np.cos(theta) + y * np.sin(theta)
    y_rot = -x * np.sin(theta) + y * np.cos(theta)

    # composante gaussienne : "fenêtre" qui localise le filtre
    gaussienne = np.exp(
        -0.5 * (x_rot ** 2 / sigma_x ** 2 + y_rot ** 2 / sigma_y ** 2)
    )

    # composante sinusoïdale : "détecteur de fréquence"
    sinusoide = np.cos(2 * np.pi * f * x_rot)

    noyau = gaussienne * sinusoide

    # centrage à zéro pour éviter une réponse DC (biais de luminosité)
    noyau -= noyau.mean()

    return noyau.astype(np.float32)


# ============================================================
#  ETAPE 3 : Application du filtre — la convolution adaptative
# ============================================================
#
# Pour chaque pixel (i, j) :
#   - si R(i,j) = 0 (hors masque) → pixel = 255 (fond blanc)
#   - si R(i,j) = 1 (dans masque) → convolution locale avec
#     le noyau Gabor construit pour θ(i,j) et f(i,j)
#
# Optimisation : on calcule UN noyau par bloc (pas par pixel),
# ce qui est une très bonne approximation car θ et f varient
# lentement à l'intérieur d'un bloc.

def appliquer_gabor(G, theta_bloc, F_bloc, freq_fiable, masque, w=16,
                    sigma_x=4.0, sigma_y=4.0, taille_noyau=11):
    """
    Applique le filtre de Gabor adaptatif à l'image normalisée G.

    G           : image normalisée uint8 (H x W)
    theta_bloc  : orientations par blocs (Hb x Wb) en radians
    F_bloc      : fréquences par blocs   (Hb x Wb) en cycles/pixel
    freq_fiable : masque booléen (Hb x Wb) — True = bloc avec mesure directe
                  sur les blocs False (interpolés = zones de singularité),
                  on recopie l'image originale plutôt que de filtrer
    masque      : masque ROI 0/255       (H x W)
    w           : taille d'un bloc

    Retourne l'image améliorée E (uint8, H x W)
    """
    H, W = G.shape
    Hb, Wb = H // w, W // w

    G_f = G.astype(np.float32)
    # initialisation : on part de l'image originale
    # → les zones douteuses seront laissées telles quelles par défaut
    E = G_f.copy()
    # fond (hors masque) = 255
    E[masque == 0] = 255.0

    demi = taille_noyau // 2

    for bi in range(Hb):
        for bj in range(Wb):

            # bornes du bloc en pixels
            y0, y1 = bi * w, min((bi + 1) * w, H)
            x0, x1 = bj * w, min((bj + 1) * w, W)

            # ⚠️  FIX 2a : si la fréquence de ce bloc est interpolée
            # (zone de singularité ou crêtes corrompues), on ne filtre pas
            # → le Gabor avec une mauvaise fréquence/orientation crée des
            #   artéfacts blancs pires que l'image originale
            if not freq_fiable[bi, bj]:
                continue   # laisse E = G_f sur ce bloc

            theta = float(theta_bloc[bi, bj])
            f     = float(F_bloc[bi, bj])

            noyau = construire_noyau_gabor(theta, f, sigma_x, sigma_y, taille_noyau)

            for i in range(y0, y1):
                for j in range(x0, x1):
                    if masque[i, j] == 0:
                        continue

                    i0 = max(0, i - demi);  i1 = min(H, i + demi + 1)
                    j0 = max(0, j - demi);  j1 = min(W, j + demi + 1)
                    fenetre = G_f[i0:i1, j0:j1]

                    ki0 = demi - (i - i0);  ki1 = ki0 + (i1 - i0)
                    kj0 = demi - (j - j0);  kj1 = kj0 + (j1 - j0)
                    k_crop = noyau[ki0:ki1, kj0:kj1]

                    E[i, j] = np.sum(fenetre * k_crop)

    # ⚠️  FIX 2b : remise à l'échelle correcte
    # Le Gabor produit des valeurs dans [-x, +x] :
    #   valeur élevée (positive)  = crête détectée   → doit devenir SOMBRE (0)
    #   valeur faible / négative  = sillon ou bruit   → doit devenir CLAIR (255)
    # On inverse donc après normalisation pour que les crêtes soient noires
    # (cohérent avec THRESH_BINARY_INV qui suit dans le pipeline)
    zone_mask = masque > 0
    zone = E[zone_mask]
    if zone.size > 0:
        e_min, e_max = zone.min(), zone.max()
        if e_max > e_min:
            # normalise dans [0, 255] puis INVERSE
            normalise = (zone - e_min) / (e_max - e_min) * 255.0
            E[zone_mask] = 255.0 - normalise   # ← inversion

    return np.clip(E, 0, 255).astype(np.uint8)


# ============================================================
#  FONCTION PRINCIPALE : gabor_enhance
# ============================================================

def gabor_enhance(img_normalisee, theta_bloc, masque, w=16,
                  sigma_x=4.0, sigma_y=4.0, taille_noyau=11, lw=32):
    """
    Pipeline complet d'amélioration par Gabor (Hong et al.).

    img_normalisee : image normalisée uint8 (512x512)
    theta_bloc     : matrice orientation (Hb x Wb) — sortie de orientation.py
    masque         : masque ROI 0/255   (512x512)  — sortie de pretraitement.py
    w              : taille d'un bloc  (doit correspondre à orientation.py)

    Retourne l'image améliorée (uint8, 512x512)
    """
    G = img_normalisee.astype(np.float32)

    print("[Gabor] Estimation de la fréquence locale...")
    F_bloc, freq_fiable = estimer_freq_image(G, theta_bloc, masque, w=w, lw=lw)

    n_fiable = int(freq_fiable.sum())
    n_total  = int((masque[::w, ::w] > 0).sum())
    print(f"[Gabor] Fréquence médiane : {np.median(F_bloc):.4f} cycles/px")
    print(f"[Gabor] Blocs fiables : {n_fiable} / {n_total} "
          f"(les autres seront laissés intacts)")

    print("[Gabor] Application du filtre...")
    E = appliquer_gabor(G, theta_bloc, F_bloc, freq_fiable, masque,
                        w=w, sigma_x=sigma_x, sigma_y=sigma_y,
                        taille_noyau=taille_noyau)
    print("[Gabor] Filtrage terminé.")
    return E

    print("[Gabor] Application du filtre (peut prendre quelques secondes)...")
    E = appliquer_gabor(G, theta_bloc, F_bloc, masque,
                        w=w, sigma_x=sigma_x, sigma_y=sigma_y,
                        taille_noyau=taille_noyau)
    print("[Gabor] Filtrage terminé.")
    return E


# ============================================================
#  EXEMPLE D'UTILISATION
# ============================================================
#
# Dans pretraitement.py, après avoir calculé l'image normalisée
# et récupéré O_bloque depuis orientation.py :
#
#   from gabor import gabor_enhance
#
#   # --- ce que tu as déjà ---
#   tab_normal = normalise_fun(image)          # image normalisée
#   masque     = masque_fun_v3(tab_normal)     # masque ROI
#   main_orientation(FILENAME)                 # calcule O_bloque.npy
#   O_bloque   = np.load('.../O_bloque.npy')   # charge les orientations
#
#   # --- la nouvelle étape Gabor ---
#   enhanced = gabor_enhance(tab_normal, O_bloque, masque, w=16)
#
#   # ensuite tu passes 'enhanced' à la place de 'contrast'
#   # dans la chaîne binarisation → morphologie → squelettisation
#
# ⚠️  La boucle pixel-par-pixel est lisible mais lente sur 512x512.
#     Pour accélérer : utiliser cv2.filter2D avec un noyau pré-calculé
#     par bloc, ou scipy.ndimage.convolve.
#     Version rapide possible si tu veux.