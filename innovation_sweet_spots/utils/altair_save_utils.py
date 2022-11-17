"""
innovation_sweet_spots.utils.altair_save_utils

Functions to save altair charts
"""
from altair_saver import save
import chromedriver_autoinstaller
import os
from typing import Iterator
from innovation_sweet_spots import PROJECT_DIR

FIGURE_PATH = f"{PROJECT_DIR}/outputs/figures"
DEFAULT_FILETYPES = ["png", "svg", "html"]


def google_chrome_driver_setup():
    """Set up the driver to save figures"""
    chromedriver_autoinstaller.install()


def create_paths(
    path: os.PathLike = FIGURE_PATH, filetypes: Iterator[list] = DEFAULT_FILETYPES
):
    """Checks if the paths exist and if not creates them"""
    for filetype in filetypes:
        os.makedirs(f"{path}/{filetype}", exist_ok=True)


def save_png(fig, path: os.PathLike, name: str, driver):
    """Save altair chart as a  raster png file"""
    save(
        fig,
        f"{path}/png/{name}.png",
        method="selenium",
        webdriver=driver,
        scale_factor=5,
    )


def save_html(fig, path: os.PathLike, name: str):
    """Save altair chart as an html file"""
    fig.save(f"{path}/html/{name}.html")


def save_svg(fig, path: os.PathLike, name: str, driver):
    """Save altair chart as a vector svg file"""
    save(fig, f"{path}/svg/{name}.svg", method="selenium", webdriver=driver)


class AltairSaver:
    """
    Class helping to easily save altair charts
    """

    def __init__(
        self,
        path: os.PathLike = FIGURE_PATH,
        filetypes: Iterator[list] = DEFAULT_FILETYPES,
    ):
        self.driver = google_chrome_driver_setup()
        self.path = path
        self.filetypes = filetypes

    def save(
        self, fig, name: str, path: os.PathLike = None, filetypes: Iterator[list] = None
    ):
        """
        Saves an altair figure in multiple formats (png, html and svg files)

        Args:
            fig: altair chart
            name: name to save the figure
            driver: webdriver
            path: path where to save the figure
            filetype: list of filetypes, eg: ['png', 'svg', 'html']
        """
        # Default values
        path = self.path if path is None else path
        filetypes = self.filetypes if filetypes is None else filetypes
        # Check paths
        create_paths(path, filetypes)
        if "png" in filetypes:
            save_png(fig, path, name, self.driver)
        if "html" in filetypes:
            save_html(fig, path, name)
        if "svg" in filetypes:
            save_svg(fig, path, name, self.driver)
