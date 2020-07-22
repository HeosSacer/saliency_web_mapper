"""
this script shows the clicking and finding a template ability of this framework
"""

from pyautomonkey import image_tools
from PIL import Image

from time import sleep
from typing import Dict
import webbrowser


class WebImageRetriever:
    def __init__(self, url: str, window_name: str = None):
        self.url = url
        self.window_name = window_name

    def retrieve_image(self):
        """
        opens an web browser from given url and gets the images specified
        """
        webbrowser.open(self.url)
        self.image, _ = image_tools.retrieve_image(self.window_name)

    def show_last_image(self):



# Load Templates
class Templates:
    def __init__(self, template_path_dict: Dict[str, str]):
        self.template_keys = template_path_dict.keys()
        for key in template_path_dict.keys():
            self.__setattr__(key, load_template(template_path_dict[key]))

    def __len__(self):
        return len(self.template_keys)

    def __iter__(self):
        self.idx = -1
        return self

    def __next__(self):
        try:
            self.idx += 1
            return self.__getattribute__(self.template_keys[self.idx])
        except IndexError:
            raise StopIteration


templates = Templates(template_dict)

print('Wait till cart UI is up...')

matching_probability = 0
print(f"xy {utils.get_mouse_pos()}")

# Wait, till game is loaded/ game logo appeared...
while matching_probability < 0.85:
    __, matching_probability = find_template(templates.product_search, window_name="Cart 4.0")
    print(f"Probs {matching_probability}")

xy = auto.click_on_template(templates.product_search, matching_threshold=0.85, window_name="Cart 4.0")
print(f"Coords {xy}")
print(f"xy {utils.get_mouse_pos()}")
sleep(2)  # Wait 2 seconds to get the gui to do its stuff
utils.click(xy)  # click on the previous coordinates
print(f"xy {utils.get_mouse_pos()}")
sleep(2)  # Wait 2 seconds to get the gui to do its stuff
