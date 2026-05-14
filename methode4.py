import numpy as np
import json

''' *************************************** OUTILS *************************************** '''

def wrap_angle(theta):
    #Ramène un angle en radians dans l'intervalle [-pi, pi].
    return (theta + np.pi) % (2 * np.pi) - np.pi

def angle_diff_biom(theta1, theta2):
    #Calcule la plus petite différence angulaire entre deux angles.
    d = abs(wrap_angle(theta1 - theta2))
    return min(d, abs(d - np.pi))

SPATIAL_TOL = 0.5
ANGLE_TOL = 0.3

''' *************************************** VOISINS *************************************** '''

_neighbors_cache = {}

def chgt_base(M, t):
    #change la base des minuties M en prenant la minutie t comme référence.
    xm, ym, theta_m, _ = M[t]
    M_new = []
    for i in range(len(M)):
        xi, yi, theta_i, typ_i = M[i]
        xi -= xm
        yi -= ym
        xtmp = xi
        xi = np.cos(theta_m) * xi + np.sin(theta_m) * yi
        yi = -np.sin(theta_m) * xtmp + np.cos(theta_m) * yi
        theta_new = wrap_angle(theta_i - theta_m)
        M_new.append((xi, yi, theta_new, typ_i))
    return M_new

def neighbors(M, nb_nghbr, i):
    #Retourne les nb_nghbr plus proches voisins de la minutie i dans M, dans la base centrée sur i.
    key = (id(M), i)
    if key in _neighbors_cache:
        return _neighbors_cache[key]

    M_loc = chgt_base(M, i)
    dists = []

    for j in range(len(M_loc)):
        if j == i:
            continue
        x, y, _, _ = M_loc[j]
        dists.append((np.hypot(x, y), j))

    dists.sort()
    res = dists[:nb_nghbr]
    _neighbors_cache[key] = res
    return res   # [(distance, index), ...]

''' *************************************** ALIGNEMENT *************************************** '''

def kabsch_umeyama(A, B):
    #Calcule la transformation affine (rotation R, échelle c, translation t) alignant B sur A.
    EA = np.mean(A, axis=0)
    EB = np.mean(B, axis=0)
    VarA = np.mean(np.linalg.norm(A - EA, axis=1)**2)
    H = ((A - EA).T @ (B - EB)) / A.shape[0]
    U, D, VT = np.linalg.svd(H)
    d = np.sign(np.linalg.det(U) * np.linalg.det(VT))
    S = np.diag([1] * (A.shape[1] - 1) + [d])
    R = U @ S @ VT
    c = VarA / np.trace(np.diag(D) @ S)
    t = EA - c * R @ EB
    return R, c, t

def apply_transformation(minutiae, R, c, t):
    #Applique la transformation affine aux minuties.
    new_minutiae = []
    angle_rot = np.arctan2(R[1, 0], R[0, 0])
    for m in minutiae:
        x, y = m["coordonnees"]
        theta = m["orientation"]
        xy_new = t + c * (R @ np.array([x, y]))
        theta_new = wrap_angle(theta + angle_rot)
        new_minutiae.append({
            "coordonnees": xy_new.tolist(),
            "orientation": theta_new,
            "type": m["type"]
        })
    return new_minutiae

''' *************************************** CLASSIFICATION PONDÉRÉE *************************************** '''

def classify_weighted(minutiae_recherche, minutiae_catalogue):
    nc = len(minutiae_catalogue)
    matched = 0
    for i in range(nc):
        xc, yc = minutiae_catalogue[i]["coordonnees"]
        dc = minutiae_catalogue[i]["orientation"]
        tc = minutiae_catalogue[i]["type"]
        for m in minutiae_recherche:
            xr, yr = m["coordonnees"]
            dr = m["orientation"]
            tr = m["type"]
            if tr != tc:
                continue
            dist = np.hypot(xr - xc, yr - yc)
            angle_diff = angle_diff_biom(dr, dc)
            if dist < SPATIAL_TOL and angle_diff < ANGLE_TOL:
                matched += 1
                break  # ← une seule correspondance suffit par minutie catalogue
    return matched, nc

def matching_score_weighted(matched, total_catalogue, total_recherche):
    if total_catalogue == 0 or total_recherche == 0:
        return 0
    precision = matched / total_recherche   # proportion de la recherche bien placée
    recall    = matched / total_catalogue   # proportion du catalogue retrouvée
    if precision + recall == 0:
        return 0
    f1 = 2 * precision * recall / (precision + recall)
    return f1 * 100

''' *************************************** MATCHING GLOBAL *************************************** '''
MIN_REF = 15
NB_NGHBR = 20
LIM_THETA = 0.4
LIM_ERREUR_DIST = 1

def global_matching_reference(data_catalogue, data_recherche):
    with open(data_catalogue, "r") as f:
        catalogue = json.load(f)
    with open(data_recherche, "r") as f:
        recherche = json.load(f)

    M_recherche = [
        (m["coordonnees"][0], m["coordonnees"][1], m["orientation"], m["type"])
        for m in recherche[0]["minutiae"] ]
    best_score = 0
    best_name = None

    for p in range(len(catalogue)):
        print("Analyse de :", catalogue[p]["nom"])
        M_catalogue = [
            (m["coordonnees"][0], m["coordonnees"][1], m["orientation"], m["type"])
            for m in catalogue[p]["minutiae"]]

        ref_r = [] 
        ref_c = []
        used_r = set() 
        used_c = set()

        for i in range(len(M_recherche)):
            for j in range(len(M_catalogue)):

                if abs(wrap_angle(M_recherche[i][2] - M_catalogue[j][2])) > LIM_THETA:
                    continue
                if M_recherche[i][3] != M_catalogue[j][3]:
                    continue

                ngh_r = neighbors(M_recherche, NB_NGHBR, i)
                ngh_c = neighbors(M_catalogue, NB_NGHBR, j)

                if len(ngh_r) != len(ngh_c):
                    continue
                dr = sorted([d for d, _ in ngh_r]) 
                dc = sorted([d for d, _ in ngh_c])
                erreur = sum(abs(dr[k] - dc[k]) for k in range(len(dr)))

                if erreur < LIM_ERREUR_DIST and i not in used_r and j not in used_c:
                    ref_r.append(M_recherche[i][:2])
                    ref_c.append(M_catalogue[j][:2])
                    used_r.add(i)
                    used_c.add(j)
        if len(ref_r) < MIN_REF:
            print("Références insuffisantes :", len(ref_r))
            continue

        R, c, t = kabsch_umeyama(np.array(ref_r), np.array(ref_c))
        minutiae_aligned = apply_transformation(recherche[0]["minutiae"], R, c, t)
        matched, total_catalogue = classify_weighted(minutiae_aligned, catalogue[p]["minutiae"])
        score = matching_score_weighted(matched, total_catalogue, len(minutiae_aligned))
        confidence = min(len(ref_r) / 20, 1.0)  # normalisé entre 0 et 1
        score = score * (0.6 + 0.4 * confidence)  # pondère le score final
        print("Score :", score)
        if score > best_score:
            best_score = score 
            best_name = catalogue[p]["nom"]

    return best_name, best_score

''' *************************************** TEST *************************************** '''

nom, score = global_matching_reference("catalogue.json", "recherche.json")
if score < 65 : 
    print("ATTENTION : Pas de correspondance fiable ! ")
print("\nMeilleure correspondance :", nom)
print("Score :", score)