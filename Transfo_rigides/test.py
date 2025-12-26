import math
import numpy as np
from itertools import combinations
import json_utils as ju 

# ----- Assure-toi que ces fonctions existent dans ton module / workspace -----
# _charger_base(data_file)
# _sauvegarder_base(new_base, data_file)
# (Si elles sont dans un autre fichier, importe-les ici.)

# ---------------------- UTILITAIRES ANGLES / VECTEURS ----------------------

def angle_normalize_pi(theta):
    """Normalise l'angle à [0, pi) comme dans l'article (theta et theta+pi équivalents)."""
    # ramène en [0, 2pi)
    t = theta % (2 * math.pi)
    # identifie theta et theta+pi -> ramène en [0, pi)
    if t >= math.pi:
        t -= math.pi
    return t

def angle_diff_pi(a, b):
    """Différence absolue d'angle en tenant compte que theta ~ theta+pi."""
    a = angle_normalize_pi(a)
    b = angle_normalize_pi(b)
    diff = abs(a - b)
    if diff > math.pi / 2:  # si > pi/2, on peut symétriser (car nous avons [0,pi) base)
        diff = abs(diff - math.pi)
    return diff

def extract_theta(m):
    """
    Convertit les orientations de ton JSON :
    - si orientation = float → on garde
    - si orientation = [ox, oy] → angle = atan2(oy, ox)
    """
    ori = m.get("orientation", None)
    if ori is None:
        # fallback si jamais le champ était différent
        ori = m.get("orient", m.get("angle", 0.0))

    if isinstance(ori, (int, float)):
        return float(ori)

    if isinstance(ori, (list, tuple)) and len(ori) == 2:
        ox, oy = ori
        return math.atan2(oy, ox)

    # Cas imprévu → pas d'orientation
    return 0.0

# ---------------------- NEIGHBOR FEATURE (ridge-based) ----------------------

def compute_neighbor_sequence(minutiae, center_idx, max_neighbors=10):
    O = minutiae[center_idx]
    Ox, Oy = O["coord"]
    theta = angle_normalize_pi(extract_theta(O))  # ✔ correction

    # vecteur horizontal (direction des crêtes)
    hx = math.cos(theta)
    hy = math.sin(theta)

    # vecteur vertical orthogonal
    vx = -hy
    vy = hx

    projections = []
    for idx, m in enumerate(minutiae):
        if idx == center_idx:
            continue

        x, y = m["coord"]
        dx = x - Ox
        dy = y - Oy

        coord_h = dx*hx + dy*hy
        coord_v = dx*vx + dy*vy

        projections.append((idx, coord_v, coord_h, math.hypot(dx, dy)))

    projections.sort(key=lambda t: (t[1], t[3]))
    sequence = [p[0] for p in projections[:max_neighbors]]
    return sequence


# ---------------------- GENERATION DE SOUS-STRUCTURES ----------------------

def generate_substructures(minutiae_test, minutiae_temp, th_theta=0.5, seq_len=6, max_neighbors=10):
    """
    Génére des paires de sous-structures candidates.
    - On parcourt toutes les paires (i,j) telles que |theta_i - theta_j| < th_theta,
      on construit la neighbor sequence autour de i (dans test) et autour de j (dans template),
      puis on crée une sous-structure de longueur seq_len (indices réels dans minutiae arrays).
    Retourne une liste de tuples (test_indices_list, temp_indices_list) chaque liste contenant
    les indices des minutiae formant la sous-structure (le premier élément est la référence).
    """
    candidates = []
    Ntest = len(minutiae_test)
    Ntemp = len(minutiae_temp)

    for i in range(Ntest):
        for j in range(Ntemp):
            if angle_diff_pi(minutiae_test[i]["orient"], minutiae_temp[j]["orient"]) < th_theta:
                # construire neighbor sequences
                seq_test = compute_neighbor_sequence(minutiae_test, i, max_neighbors=max_neighbors)
                seq_temp = compute_neighbor_sequence(minutiae_temp, j, max_neighbors=max_neighbors)
                # insérer le centre en tête (optionnel mais utile)
                seq_test_full = [i] + seq_test
                seq_temp_full = [j] + seq_temp
                # ne garder que seq_len éléments (si possible)
                if len(seq_test_full) >= seq_len and len(seq_temp_full) >= seq_len:
                    candidates.append((seq_test_full[:seq_len], seq_temp_full[:seq_len]))
    return candidates

# ---------------------- TEST DE CORRESPONDANCE PAR CLASSIFIER LINES ----------------------

def sign_of_minutia_wrt_line(p1, p2, q):
    """
    Pour une line passant par p1->p2, calcule le 'signe' de q comme dans l'article.
    - On projette q sur la ligne p1p2 (point R)
    - vecteur v_line = vecteur unité le long de p1->p2
    - vecteur v_proj = q - R
    - signe = sign( cross(v_line, v_proj) )  => +1 / -1 / 0
    """
    p1 = np.asarray(p1); p2 = np.asarray(p2); q = np.asarray(q)
    line = p2 - p1
    line_norm = np.linalg.norm(line)
    if line_norm == 0:
        return 0
    v_line = line / line_norm
    # projection param t along line: R = p1 + t * v_line
    t = np.dot(q - p1, v_line)
    R = p1 + t * v_line
    v_proj = q - R
    # cross product z-component (2D)
    cross_z = v_line[0] * v_proj[1] - v_line[1] * v_proj[0]
    if cross_z > 1e-8:
        return 1
    elif cross_z < -1e-8:
        return -1
    else:
        return 0

def substructure_correspond(s_test, s_temp, minutiae_test, minutiae_temp, require_all_lines=True):
    """
    Vérifie si deux sous-structures (listes d'indices) sont correspondantes suivant la méthode
    du classifier line : pour toutes paires de points dans la sous-structure, la partition des autres
    points en classes (signes) doit correspondre.
    - s_test, s_temp : listes d'indices (même longueur)
    - retourne True si au moins un classifier line correspondant produit le même pattern.
    Le param require_all_lines contrôle la sévérité (True => tous les classifier lines doivent correspondre).
    """
    c = len(s_test)
    if c != len(s_temp):
        return False

    # On teste paires de référence (u,v) dans la sous-structure.
    # Pour robustesse, on demandera qu'au moins un paire (u,v) donne correspondance identique.
    for (a_idx, b_idx) in combinations(range(c), 2):
        # classifier line in test: between s_test[a_idx] and s_test[b_idx]
        p_test_a = minutiae_test[s_test[a_idx]]["coord"]
        p_test_b = minutiae_test[s_test[b_idx]]["coord"]
        # classifier line in temp: between s_temp[a_idx] and s_temp[b_idx]
        p_temp_a = minutiae_temp[s_temp[a_idx]]["coord"]
        p_temp_b = minutiae_temp[s_temp[b_idx]]["coord"]

        # compute signs for remaining points in test
        signs_test = []
        for k in range(c):
            idx = s_test[k]
            signs_test.append(sign_of_minutia_wrt_line(p_test_a, p_test_b, minutiae_test[idx]["coord"]))

        signs_temp = []
        for k in range(c):
            idx = s_temp[k]
            signs_temp.append(sign_of_minutia_wrt_line(p_temp_a, p_temp_b, minutiae_temp[idx]["coord"]))

        # Compare signs pattern (we consider sign 0 equal to either +1/-1? article le considère binaire)
        # Ici on compare exactement les signes. On tolère petites différences en autorisant 0 ~ +1 ou -1 if needed.
        if signs_test == signs_temp:
            # paire de classifier lines correspondant -> sous-structure correspond
            return True
        # sinon continue
        if require_all_lines:
            # si on exige que toutes les paires correspondent, on continue et ne retourne que si toutes ok
            continue

    # si aucune paire acceptable trouvée
    return False

# ---------------------- ESTIMATION TRANSFORMATION POLYNOMIALE ----------------------

def polynomial_terms_2d(x, y, d):
    """
    Retourne la liste des monomes 1, x, y, x^2, xy, y^2, ..., jusqu'à degré d
    dans un ordre fixe (p from 0..d, q from 0..(d-p))
    """
    terms = []
    for p in range(d+1):
        for q in range(d+1-p):
            terms.append((p, q))
    vals = [ (x**p) * (y**q) for (p,q) in terms ]
    return vals

def estimate_polynomial_transform(src_pts, dst_pts, d=2):
    """
    Estime deux vecteurs de coefficients a_x et a_y pour la transformation polynomiale:
      x* = sum a_x_k * monome_k(x,y)
      y* = sum a_y_k * monome_k(x,y)
    src_pts, dst_pts : Nx2 numpy arrays
    Retourne coeffs_x, coeffs_y, terms_list
    """
    src_pts = np.asarray(src_pts)
    dst_pts = np.asarray(dst_pts)
    N = src_pts.shape[0]
    # nombre de monomes
    M = (d+1)*(d+2)//2
    if N < M:
        raise ValueError(f"Pas assez de points pour estimer un polynôme d'ordre {d}. Nécessite >= {M} pts, obtenu {N}.")

    # Construire matrice A (N x M) design pour monomes; on fera concat pour x et y indépendants
    A = np.zeros((N, M))
    for i in range(N):
        x, y = src_pts[i]
        A[i, :] = polynomial_terms_2d(x, y, d)

    bx = dst_pts[:, 0]
    by = dst_pts[:, 1]

    # solution least squares
    coeffs_x, *_ = np.linalg.lstsq(A, bx, rcond=None)
    coeffs_y, *_ = np.linalg.lstsq(A, by, rcond=None)

    # Construire la liste de (p,q) pour interprétation
    terms = []
    for p in range(d+1):
        for q in range(d+1-p):
            terms.append((p, q))

    return coeffs_x, coeffs_y, terms

def apply_polynomial_transform(minutiae, coeffs_x, coeffs_y, terms):
    """
    Applique la transformation polynomiale à la liste de minutiae.
    minutiae : liste de dicts { "coord":[x,y], "orient":theta }
    retourne nouvelle liste transformée (coord + orient inchangé)
    """
    out = []
    for m in minutiae:
        x, y = m["coord"]
        mon = [ (x**p)*(y**q) for (p,q) in terms ]
        x_new = float(np.dot(coeffs_x, mon))
        y_new = float(np.dot(coeffs_y, mon))
        out.append({"coord":[x_new, y_new], "orient": m["orient"]})
    return out

# ---------------------- EXEMPLE D'UTILISATION : pipeline pour 2 empreintes ----------------------

def compute_transforms_between_two(minutiae_test_json, minutiae_temp_json,
                                   th_theta=0.5, seq_len=6, max_neighbors=10, poly_order=2):
    """
    Pipeline:
    - lit les minutiae (au format JSON dictionary contenant 'minutiae' champ)
    - génère sous-structures candidates
    - filtre par méthode classifier lines (substructure_correspond)
    - pour chaque sous-structure correspondante, estime une transform polynomiale d'ordre poly_order
    Retourne une liste de transformations valides (coeffs_x, coeffs_y, terms) triées par nombre de points utilisés (plus grand d'abord).
    """
    # convertir JSON -> liste de minutiae standardisée
    def to_list(min_json):
        L = []
        for m in min_json["minutiae"]:
            # orientation peut être simple nombre ou paire; on garde le premier si liste
            ori = m["orientation"]
            if isinstance(ori, (list, tuple)):
                ori = ori[0]
            L.append({"coord": m["coordonnees"], "orient": float(ori)})
        return L

    test = to_list(minutiae_test_json)
    temp = to_list(minutiae_temp_json)

    sub_cands = generate_substructures(test, temp, th_theta=th_theta, seq_len=seq_len, max_neighbors=max_neighbors)

    transforms = []
    for s_test, s_temp in sub_cands:
        # Vérifier correspondance via classifier lines
        if not substructure_correspond(s_test, s_temp, test, temp, require_all_lines=False):
            continue
        # extraire coordonnées correspondantes (src -> test, dst -> temp)
        src_pts = np.array([ test[idx]["coord"] for idx in s_test ])
        dst_pts = np.array([ temp[idx]["coord"] for idx in s_temp ])
        try:
            coeffs_x, coeffs_y, terms = estimate_polynomial_transform(src_pts, dst_pts, d=poly_order)
            transforms.append((coeffs_x, coeffs_y, terms, len(s_test)))
        except ValueError:
            # pas assez de points pour cet ordre => ignorer
            continue

    # trier par nombre de points (desc)
    transforms.sort(key=lambda t: t[3], reverse=True)
    return transforms

# ---------------------- UTILISATION SUR UNE BASE (catalogue / recherche) ----------------------

def find_transforms_for_search(catalogue_file, recherche_file, th_theta=0.5, seq_len=6, max_neighbors=10, poly_order=2):
    """
    Charge le catalogue (liste d'entrées JSON) et la recherche (on prend la première entrée du fichier recherche),
    puis pour chaque entrée du catalogue cherche des transforms candidates.
    Retourne dict: nom_personne -> list of transforms
    """
    base = ju._charger_base(catalogue_file)
    recherche_list = ju._charger_base(recherche_file)
    if not recherche_list:
        raise ValueError("fichier recherche vide")
    recherche = recherche_list[0]

    results = {}
    for person in base:
        transforms = compute_transforms_between_two(recherche, person,
                                                   th_theta=th_theta, seq_len=seq_len,
                                                   max_neighbors=max_neighbors, poly_order=poly_order)
        results[person["nom"]] = transforms
    return results



# -----------------------------------------------------------
# SCORE DE MATCH FINAL ENTRE DEUX EMPREINTES ALIGNEES
# -----------------------------------------------------------

def match_score(minA, minB, th_dist=15.0, th_theta=0.5):
    """
    Compare deux listes de minutiae APRES ALIGNEMENT.
    Renvoie un score entier = nombre de minutiae correspondantes.
    """
    score = 0
    for m1 in minA:
        x1, y1 = m1["coord"]
        t1 = m1["orient"]
        for m2 in minB:
            x2, y2 = m2["coord"]
            t2 = m2["orient"]

            # distance spatiale
            dx, dy = x1 - x2, y1 - y2
            if math.hypot(dx, dy) < th_dist:
                # différence d’orientation
                if angle_diff_pi(t1, t2) < th_theta:
                    score += 1
    return score


# -----------------------------------------------------------
# PIPELINE COMPLET : trouver transformation -> l'appliquer -> comparer
# -----------------------------------------------------------

def compare_empreintes_with_transform(test_json, temp_json,
                                      th_theta=0.5, seq_len=6,
                                      max_neighbors=10, poly_order=2,
                                      th_dist=15.0):
    """
    Pipeline complet :
    1. trouve les transformations candidates (polynôme)
    2. applique la meilleure transformation
    3. calcule le score final de matching

    test_json : empreinte recherchée (dict)
    temp_json : empreinte du catalogue (dict)
    """
    # 1) TROUVER LES TRANSFORMATIONS
    transforms = compute_transforms_between_two(test_json, temp_json,
                                                th_theta=th_theta,
                                                seq_len=seq_len,
                                                max_neighbors=max_neighbors,
                                                poly_order=poly_order)

    if not transforms:
        return 0  # aucune transformation trouvée → aucun match possible

    # meilleure transformation = celle avec le plus de points de correspondance
    coeffs_x, coeffs_y, terms, npts = transforms[0]

    # 2) APPLIQUER LA TRANSFORMATION A L'EMPREINTE TEST
    def convert(json_data):
        L = []
        for m in json_data["minutiae"]:
            ori = m["orientation"]
            if isinstance(ori, (list, tuple)):
                ori = ori[0]
            L.append({"coord": m["coordonnees"], "orient": float(ori)})
        return L

    test_list = convert(test_json)
    temp_list = convert(temp_json)

    test_transformed = apply_polynomial_transform(test_list, coeffs_x, coeffs_y, terms)

    # 3) COMPARER LES MINUTIAE ALIGNÉES
    score = match_score(test_transformed, temp_list,
                        th_dist=th_dist, th_theta=th_theta)

    return score


def comparer_catalogue(catalogue_file, recherche_file):
    base = ju._charger_base(catalogue_file)
    recherche_list = ju._charger_base(recherche_file)

    if not recherche_list:
        raise ValueError("Fichier recherche vide.")

    recherche = recherche_list[0]

    résultats = []
    for person in base:
        score = compare_empreintes_with_transform(recherche, person)
        résultats.append((person["nom"], score))

    # On trie du meilleur score au pire
    résultats.sort(key=lambda t: t[1], reverse=True)
    return résultats


scores = comparer_catalogue("catalogue.json", "recherche.json")

for nom, score in scores:
    print(f"{nom} → score = {score}")
