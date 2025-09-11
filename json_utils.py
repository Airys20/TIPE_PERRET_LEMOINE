import json
data_file = 'base.json'

#FONCTION GESTION DE BASE 

def _charger_base():
    if os.path.exists(data):
        with open(data_file, "r", encoding="utf-8") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return []
    return []

def _sauvegarder_base(base):
    with open(data_file, "w", encoding="utf-8") as f:
        json.dump(base, f, indent=4, ensure_ascii=False)

###############################################################################################

def ajouter_personne(nom_personne, minutiae_tab):
    """
    struct de minutiae_tab :
    [
        [ [x,y], "type", [x,y] ],\n
        [ [x,y], "type", [x,y] ], \n
        ---

    ]
    """

    base = _charger_base()
    # modif format pour coller au json
    minutiae_struct = []
    for elt in minutiae_tab:
        coord, typ, orient = elt
        minutiae_struct.append({
            "coordonnees": coord,
            "type": typ,
            "orientation": orient
        })

    
    nouvelle_entree = {
        "nom": nom_personne,
        "minutiae": minutiae_struct
    }

    # add+ sauvegarde!!!!!
    base.append(nouvelle_entree) 

    _sauvegarder_base()
    print(f"add {nom_personne}")



"""
test
 input = [
    [[10, 20], "bifurcation", [1.5, 2.5]],
    [[15, 25], "endings", [2.0, 3.0]],

]

ajouter_personne("p6", input)

"""

def modifier_personne(nom_personne, minutiae_tab=None, nouveau_nom=None, merge=False): #qd None -> PEUT etre remplacé mais Pas OBLIGE
    base = _charger_base()
    index = None  
    for i, p in enumerate(base):
        if p.get("nom") == nom_personne:
            index = i
            break  #s'arrete si trouve
    if index is None:
        raise ValueError(f"'{nom_personne}' introuvable")
    if minutiae_tab is not None:
        nouvelles = []
        for item in minutiae_tab:
            coord = item[0]
            typ = item[1]
            orient = item[2]

            element = {
                "coordonnees": coord,
                "type": typ,
                "orientation": orient
            }
            nouvelles.append(element)


