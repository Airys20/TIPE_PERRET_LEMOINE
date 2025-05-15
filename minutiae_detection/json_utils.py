import json

def write(id, tab_coord,type,tab_orientation):
# Data to be written
    dictionary = {
            "nom": id,
            "minutiae": [
                {
                    "coordonées": tab_coord,
                    "type": type,
                    "orientation": tab_orientation

                }
            ]

        },

    with open('minutiae_detection\\base.json', 'a', encoding='utf-8') as f:
        json.dump(dictionary, f, ensure_ascii=False, indent=4)

write("p4", [5,7],"endings",[7,5])