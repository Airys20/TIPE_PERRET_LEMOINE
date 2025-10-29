import numpy as np
import json_utils as ju

# Trouver le centre de l'empreinte
def center(Matrix):
    [x, y] = np.mean(Matrix, axis=0)
    return (x, y)


def new_cat(pers, data1, data2, n_minutiae=1000):
    """
    Extrait les n_minutiae minuties les plus proches du centre
    et les ajoute dans un nouveau catalogue.
    
    pers : index de la personne dans le catalogue d'origine (int)
    data1 : fichier JSON d'origine
    data2 : fichier JSON de destination
    n_minutiae : nombre de minuties à conserver (par défaut 75)
    """
    
    # Récupération des données de la personne
    data_personne = ju.get_data(pers, data1)
    M_dico = data_personne["minutiae"]
    
    # Conversion en tableau numpy des coordonnées
    M_list = [m['coordonnees'] for m in M_dico]
    M = np.array(M_list)
    
    # Calcul du centre
    (x0, y0) = center(M)
    
    # Calcul des distances de chaque minutie au centre
    distances = np.sqrt((M[:,0] - x0)**2 + (M[:,1] - y0)**2)
    
    # Tri selon la distance croissante
    indices_tries = np.argsort(distances)
    
    # Nombre de minuties à garder (si < n dispo)
    n_minutiae = min(n_minutiae, len(indices_tries))
    
    # Construction du tableau des minuties retenues
    minutiae_tab = []
    for i in range(n_minutiae):
        idx = indices_tries[i]
        minutiae_tab.append([
            M_dico[idx]['coordonnees'],
            M_dico[idx]['type'],
            M_dico[idx]['orientation']
        ])
    
    # Sauvegarde dans le nouveau catalogue
    ju.ajouter_personne(data_personne["nom"], minutiae_tab, data2)
    print(f"{n_minutiae} minuties les plus proches ajoutées à {data2}")
    return


def distance(M, n_minutiae=1000):
    """
    Retourne une nouvelle matrice contenant les n_minutiae points
    les plus proches du centre de M.
    """
    (x0, y0) = center(M)
    
    distances = np.sqrt((M[:,0] - x0)**2 + (M[:,1] - y0)**2)
    indices_tries = np.argsort(distances)
    
    n_minutiae = min(n_minutiae, len(indices_tries))
    new_matrix = M[indices_tries[:n_minutiae]]
    
    return new_matrix

