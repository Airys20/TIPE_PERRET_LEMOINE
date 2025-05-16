import json
data_file = 'base.json'


def ajouter_personne(nom_personne, minutiae_compact):
    """
    struct de minutiae_compact :
    [
        [ [x,y], "type", [x,y] ],\n
        [ [x,y], "type", [x,y] ], \n
        ---

]
    
    """
    with open(data_file, "r", encoding="utf-8") as f:
        base = json.load(f)
    
    # modif format 
    minutiae_struct = []
    for item in minutiae_compact:
        coord, typ, orient = item
        minutiae_struct.append({
            "coordonées": coord,
            "type": typ,
            "orientation": orient
        })

    # new enttry
    nouvelle_entree = {
        "nom": nom_personne,
        "minutiae": minutiae_struct
    }

    # Ajout et sauvegarde
    base.append(nouvelle_entree)
    with open(data_file, "w", encoding="utf-8") as f:
        json.dump(base, f, indent=4, ensure_ascii=False)

    print(f"Entrée ajoutée pour {nom_personne}.")



minutiae_input = [
    [[10, 20], "bifurcation", [1.5, 2.5]],
    [[15, 25], "endings", [2.0, 3.0]],

]

ajouter_personne("p6", minutiae_input)
