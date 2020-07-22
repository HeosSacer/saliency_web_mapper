"""
this script shows the clicking and finding a template ability of this framework
"""

from pyautomonkey import image_tools
from PIL import Image

from time import sleep
import webbrowser


class WebImageRetriever:
    def __init__(self, url: str, window_name: str = None):
        self.url = url
        self.window_name = window_name
        self.image: Image = None

    def retrieve_image(self):
        """
        opens an web browser from given url and gets the images specified
        """
        webbrowser.open(self.url)
        sleep(3)
        _, self.image = image_tools.retrieve_image(self.window_name)

    def show_last_image(self):
        self.image.show()
