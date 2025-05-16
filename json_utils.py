import json
data_file = 'base.json'


def ajouter_personne(nom_personne, minutiae_tab):
    """
    struct de minutiae_tab :
    [
        [ [x,y], "type", [x,y] ],\n
        [ [x,y], "type", [x,y] ], \n
        ---

]
    
    """


    with open(data_file, "r", encoding="utf-8") as f:
        base = json.load(f) #ouvre base
    
    # modif format pour coller au json
    minutiae_struct = []
    for elt in minutiae_tab:
        coord, typ, orient = elt
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

    # add+ sauvegarde
    base.append(nouvelle_entree) #(add)

    with open(data_file, "w", encoding="utf-8") as f:
        json.dump(base, f, indent=4, ensure_ascii=False) #dump = l'ajoute et sauvegarde dans le fichierc ?

    print(f"add {nom_personne}") #check


"""
test
 input = [
    [[10, 20], "bifurcation", [1.5, 2.5]],
    [[15, 25], "endings", [2.0, 3.0]],

]

ajouter_personne("p6", input)

"""
