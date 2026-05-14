
import json
import numpy as np

with open("catalogue.json") as f:
    cat = json.load(f)
with open("recherche.json") as f:
    rech = json.load(f)

m_cat = cat[0]["minutiae"]
m_rech = rech[0]["minutiae"]

print("=== CATALOGUE (première empreinte) ===")
print("Nombre de minuties :", len(m_cat))
print("Exemples :", m_cat[:2])
xs = [m["coordonnees"][0] for m in m_cat]
ys = [m["coordonnees"][1] for m in m_cat]
print(f"X : [{min(xs):.2f}, {max(xs):.2f}]  Y : [{min(ys):.2f}, {max(ys):.2f}]")

print("\n=== RECHERCHE ===")
print("Nombre de minuties :", len(m_rech))
print("Exemples :", m_rech[:2])
xs = [m["coordonnees"][0] for m in m_rech]
ys = [m["coordonnees"][1] for m in m_rech]
print(f"X : [{min(xs):.2f}, {max(xs):.2f}]  Y : [{min(ys):.2f}, {max(ys):.2f}]")
with open("recherche.json") as f:
    rech = json.load(f)

print("Nombre d'entrées dans recherche.json :", len(rech))
for i, entry in enumerate(rech):
    print(f"  [{i}] minuties : {len(entry['minutiae'])}")

bbox = intersection_bbox(minutiae_aligned, catalogue[p]["minutiae"])
if bbox:
    n1 = len(filter_in_bbox(minutiae_aligned, bbox))
    n2 = len(filter_in_bbox(catalogue[p]["minutiae"], bbox))
    print(f"  Dans intersection : N1={n1}, N2={n2}, bbox={bbox}")
    