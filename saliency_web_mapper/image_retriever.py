"""
this script shows the clicking and finding a template ability of this framework
"""

from pyautomonkey import image_tools
from PIL import Image
from time import sleep, time
import webbrowser


class WebImageRetriever:
    def __init__(self, url: str, window_name: str = None):
        self.url = url
        self.window_name = window_name
        self.image: Image = None

    def image_retrieve_session(self, timeout: float = None, period: float = 1):
        """
        saves images each period
        """
        im_cnt = 0
        im_list = []

        if timeout:
            t = time()

        while True:
            if timeout:
                if time() - t > timeout:
                    break
            sleep(period)
            im: Image = self.retrieve_image()
            #self.save_image(im, f"/session/{im_cnt}.png")
            im_cnt += 1
            im_list.append(im)

        return im_list

    def save_image(self, im: Image, path: str):
        im.save(path, "PNG")

    def retrieve_image(self) -> Image:
        """
        opens an web browser from given url and gets the images specified
        """
        webbrowser.open(self.url)
        sleep(3)
        _, self.image = image_tools.retrieve_image(self.window_name)
        return self.image

    def show_last_image(self):
        self.image.show()
