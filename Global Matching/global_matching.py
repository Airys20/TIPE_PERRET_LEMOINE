import json

data_catalogue = 'catalogue.json'
data_empreinte = 'recherche.json'

# --- Comparaison minutie par minutie ---
def classify(personne):
    with open(data_catalogue, "r", encoding="utf-8") as f:
        catalogue = json.load(f)
    with open(data_empreinte, "r", encoding="utf-8") as f2:
        empreinte = json.load(f2)

    dico_personne = empreinte[0]["minutiae"]   # empreinte recherchée
    dico_catalogue = catalogue[personne]["minutiae"]

    n = len(dico_personne)
    np = len(dico_catalogue)
    paired = [0 for _ in range(n)]

    for i in range(n):  # pour chaque minutie de la recherche
        coord_pers = dico_personne[i]["coordonnees"]
        dir_pers = dico_personne[i]["orientation"]

        for j in range(np):  # on parcourt toutes celles du catalogue
            coord_cat = dico_catalogue[j]["coordonnees"]
            dir_cat = dico_catalogue[j]["orientation"]

            # Tolérance spatiale
            if abs(coord_pers[0] - coord_cat[0]) < 5 and abs(coord_pers[1] - coord_cat[1]) < 5:
                # Tolérance directionnelle
                if abs(dir_pers - dir_cat) < 0.5:
                    paired[i] = 2  # matched
                else:
                    paired[i] = 1  # juste apparié
                break  # on a trouvé un match pour cette minutie

    return paired


def matching_score( i, paired):
    with open(data_catalogue, "r", encoding="utf-8") as f2:
        empreinte = json.load(f2)
    dico_catalogue = empreinte[i]["minutiae"]

    np = len(dico_catalogue)
    Npair = paired.count(1)
    Nmatch = paired.count(2)

    return ((Nmatch +0.5*Npair) / np) * 100


def global_matching(data_file):
    """
    Parcourt le catalogue aligné pour calculer le score de matching et retourner le nom
    de la meilleure correspondance.
    """
    with open(data_file, "r", encoding="utf-8") as f:
        base = json.load(f)

    nc = len(base)
    tab_score = [0 for _ in range(nc)]
    i_max = 0

    for i in range(nc):
        paired = classify(i)
        tab_score[i] = matching_score(i, paired)
        name = base[i]["nom"]
        print(f"{name} : {tab_score[i]:.2f}")

        if tab_score[i] > tab_score[i_max]:
            i_max = i

    nom = base[i_max]["nom"]
    return nom



# 2. Lancer le matching
best_match = global_matching("catalogue.json")
print("\nMeilleure correspondance :", best_match)
