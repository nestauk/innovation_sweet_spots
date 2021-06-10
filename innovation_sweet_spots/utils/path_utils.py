import os

from createch import PROJECT_DIR


def make_dir(dir):
    if os.path.exists(f"{PROJECT_DIR}/{dir}") is False:
        os.mkdir(f"{PROJECT_DIR}/{dir}")
