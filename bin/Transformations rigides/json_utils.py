                        import json
                        import os


                        #FONCTION GESTION DE BASE 

                        def _charger_base(data_file):
                            if os.path.exists(data_file):
                                with open(data_file, "r", encoding="utf-8") as f:
                                    try:
                                        return json.load(f)
                                    except json.JSONDecodeError:
                                        return []
                            return []

                        def _sauvegarder_base(new_base,data_file):
                            with open(data_file, "w", encoding="utf-8") as f:
                                json.dump(new_base, f, indent=4, ensure_ascii=False)


                        def clear_base(data_file):
                            _sauvegarder_base([],data_file)
                            




                        ###############################################################################################

                        def ajouter_personne(nom_personne, minutiae_tab,data_file):
                            """
                            struct de minutiae_tab :
                            [
                                [ [x,y], "type", [x,y] ],\n
                                [ [x,y], "type", [x,y] ], \n
                                ---

                            ]
                            """

                            base = _charger_base(data_file)
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

                            _sauvegarder_base(base,data_file)
                            print(f"add {nom_personne}")


                        def modifier_personne(data_file, nom_personne, minutiae_tab=None, nouveau_nom=None, merge=False): #qd None -> PEUT etre remplacé mais Pas OBLIGE
                            base = _charger_base(data_file)
                            index = None  

                            for i, p in enumerate(base):

                                if p.get("nom") == nom_personne:
                                    index = i
                                    break  #s'arrete si trouve

                            if index is None:
                                raise ValueError(f"'{nom_personne}' introuvable")
                            
                            if minutiae_tab is not None: #si nouv coord
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

                            if merge:
                                base[index]["minutiae"].extend(nouvelles) #on ajoute les nouvelles coord
                            else:
                                base[index]["minutiae"] = nouvelles #on change celle qui y sont 

                            # renomme ?
                            if nouveau_nom:

                                # vérif persoonne avec ce nom
                            #[ ] Verif que personne avec ce nom
                                base[index]["nom"] = nouveau_nom
                            print(f"modif de {nom_personne}")
                            _sauvegarder_base(base, data_file)
                            return base[index]  # check enregistremt


                        #########################################################################################################################
                        #ACCESSEURS

                        def get_data(nbr_empreinte, nom_base):
                            base = _charger_base(nom_base) 
                            dico_personne = base[nbr_empreinte]
                            return dico_personne

                        def iter_coordonnees (nom_base, f, i):
                            base = _charger_base(nom_base)
                            for m in range (len(base[i]["minutiae"])):
                                base[i]["minutiae"][m]["coordonnees"] = f(base[i]["minutiae"][m]["coordonnees"]) #on change celle qui y sont 

                            
                            print(f"fonction applique a toute les coordonées de  {nom_base}")
                            _sauvegarder_base(base, nom_base)
                            

                            



                        #####################################################################################################################
                        #test
                        """
                        nom_data = 'base.json'
                        clear_base(nom_data)


                        input = [
                            [[10, 20], "bifurcation", [1.5, 2.5]],
                            [[15, 25], "endings", [2.0, 3.0]],

                        ]

                        ajouter_personne("p6", input, nom_data)

                        input2 = [
                            [[10, 20], "bifurcation", [1.5, 2.5]],
                            [[15, 25], "endings", [2.0, 3.0]],

                        ]

                        ajouter_personne("p7", input2, nom_data)

                        modifier_personne(nom_data, "p6",[
                            [[0, 0], "endings", [1.5, 2.5]],
                            [[0, 0], "endings", [2.0, 3.0]],
                        ],
                        "test_modif",False)

                        def f(m) :
                            return [m[0]+2,m[1]+10]

                        iter_coordonnees (nom_data, f)

                        """