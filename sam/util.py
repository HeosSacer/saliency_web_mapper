import os
from scipy.io import loadmat
from PIL import Image

path = "data/fixations/train/"

maps = [path + f for f in os.listdir(path) if f.endswith('.mat')]

for map_path in maps:
    mat = loadmat(map_path)
    im = Image.fromarray(mat)
    im.save(map_path + ".png" )
