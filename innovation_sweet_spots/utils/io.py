# Generic read / save functions

import json

from innovation_sweet_spots import PROJECT_DIR


def save_lookup(name, path_name):
    with open(f"{PROJECT_DIR}/{path_name}.json", "w") as outfile:
        json.dump(name, outfile)


def get_lookup(path_name):
    with open(f"{PROJECT_DIR}/{path_name}.json", "r") as infile:
        return json.load(infile)
