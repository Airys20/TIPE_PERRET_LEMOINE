import json_utils as ju
import numpy as np 
import shutil
import bin.selection_des_minuties as am
import cv2
import math
import matplotlib.pyplot as plt


def kabsch_umeyama(A, B):  # article : Aligning point patterns with Kabsch–Umeyama algorithm by Tuomas Siipola
    assert A.shape == B.shape
    n, m = A.shape

    EA = np.mean(A, axis=0)
    EB = np.mean(B, axis=0)
    VarA = np.mean(np.linalg.norm(A - EA, axis=1) ** 2)

    H = ((A - EA).T @ (B - EB)) / n
    U, D, VT = np.linalg.svd(H)
    d = np.sign(np.linalg.det(U) * np.linalg.det(VT))
    S = np.diag([1] * (m - 1) + [d])

    R = U @ S @ VT
    c = VarA / np.trace(np.diag(D) @ S)
    t = EA - c * R @ EB

    return R, c, t


def transformation_rigide(catalogue, recherche, n_minutiae=1000):
    """
    Applique une transformation rigide (rotation, translation, scaling)
    sur les empreintes du catalogue pour les aligner avec celle du fichier recherche.
    """
    # On repart d’un nouveau catalogue vide
    ju.clear_base("new_catalogue.json")

    # --- Lecture et sélection des minuties de l'empreinte de référence
    Mr_dico = ju.get_data(0, recherche)["minutiae"]
    Mr_list = [m['coordonnees'] for m in Mr_dico]
    Mr_matrix = np.array(am.distance(np.array(Mr_list), n_minutiae))

    # --- Parcours des empreintes du catalogue
    base_c = ju._charger_base(catalogue)
    n = len(base_c)

    for i in range(n):
        # On crée une version filtrée (minuties proches du centre)
        am.new_cat(i, catalogue, "new_catalogue.json", n_minutiae=n_minutiae)
        Mc_dico = ju.get_data(i, "new_catalogue.json")["minutiae"]
        Mc_list = [m['coordonnees'] for m in Mc_dico]
        Mc_matrix = np.array(Mc_list)

        # --- Création d’appariements robustes : pour chaque minutie de Mr, 
        # on cherche dans Mc la plus proche du même type
        Mr_aligned = []
        Mc_aligned = []

        for mr in Mr_dico:
            p= np.array(mr["coordonnees"])
            typ = mr["type"]

        # On ne considère que les minuties du même type
        same_type = [m for m in Mc_dico if m["type"] == typ]
        if not same_type:
            continue

        Mc_same = np.array([m["coordonnees"] for m in same_type])
        distances = np.linalg.norm(Mc_same - p, axis=1)
        j = np.argmin(distances)
        closest = Mc_same[j]

        Mr_aligned.append(p)
        Mc_aligned.append(closest)

        # conversion en array numpy
        Mr_aligned = np.array(Mr_aligned)
        Mc_aligned = np.array(Mc_aligned)
        
        # --- Calcul de la transformation qui aligne Mc sur Mr
        R, c, t = kabsch_umeyama(Mc_aligned, Mr_aligned)

        
        print(f"Empreinte {i} :")
        print("Rotation :\n", R)
        print("Échelle :", c)
        print("Translation :", t, "\n")

        # --- Application de la transformation aux coordonnées et orientations
        def f(m):
            """Transformation des coordonnées (translation + rotation + scaling)."""
            return (t + c * R @ np.array(m)).tolist()

        ju.iter_coordonnees("new_catalogue.json", f, i)

        # --- Mise à jour des orientations
        base_new = ju._charger_base("new_catalogue.json")
        minutiae = base_new[i]["minutiae"]

        for m in minutiae:
            orient = np.array(m["orientation"])
            m["orientation"] = (R @ orient).tolist()

        ju._sauvegarder_base(base_new, "new_catalogue.json")

    print("\n Transformation rigide terminée — catalogue sauvegardé dans 'new_catalogue.json'\n")
    return


def affichage(M, i, filename):
    """
    Affiche les minuties sur une image, avec leur type et leur position.
    """
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    color_image = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    for m in M:
        [x, y] = m["coordonnees"]
        typ = m["type"]
        [dx, dy] = m["orientation"]

        if typ == "ending":
            color = (0, 0, 255)  # rouge
        elif typ == "bifurcation":
            color = (0, 255, 0)  # vert
        else:
            color = (0, 255, 255)  # jaune

        cv2.circle(color_image, (int(x), int(y)), 1, color, 1)

        # --- Optionnel : tracer la direction
        lx = int(round(x + dx * 6))
        ly = int(round(y + dy * 6))
        cv2.line(color_image, (int(x), int(y)), (lx, ly), (255, 0, 0), 1)

    output_filename = f"affichage_{i}.png"
    cv2.imwrite(output_filename, color_image)

    plt.figure(figsize=(10, 10))
    plt.imshow(color_image[..., ::-1])  # BGR → RGB pour affichage matplotlib
    plt.title(f"Empreinte {i}")
    plt.axis("off")
    plt.show()


# Exemple d’utilisation :
transformation_rigide('catalogue.json', 'recherche.json', n_minutiae=1000)
M = ju.get_data(2, 'new_catalogue.json')["minutiae"]
affichage(M, 2, 'Global Matching\\Tests\\justine_output.jpg')



