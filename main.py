from methode2 import global_matching_reference
from minutiae_detection.main_find import main_find
from json_utils import clear_base

FILENAME = 'minutiae_detection\input\Syria1.jpeg'
NAME = 'Syria.1'


BASE = 'recherche.json'
data_catalogue = 'catalogue.json'
main_find(NAME, BASE, FILENAME)
global_matching_reference(data_catalogue, BASE)

clear_base(BASE)