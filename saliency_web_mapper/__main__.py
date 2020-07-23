from saliency_web_mapper.image_retriever import WebImageRetriever
from saliency_web_mapper.config.environment import SaliencyWebMapperEnvironment
import cv2
from PIL import Image
import numpy as np

def app(env: SaliencyWebMapperEnvironment):
    #images = WebImageRetriever(url=env.url, window_name=env.window_name).image_retrieve_session(timeout=20)
    im = Image.open("sam/data/images/Capture.png")
    im.show()
    im = cv2.imread("sam/data/images/Capture.png")
    #im = np.array(im.convert('RGB'))
    #im = im.astype(np.uint8)
    sal = cv2.saliency.StaticSaliencyFineGrained_create()
    (success, saliencyMap) = sal.computeSaliency(im)
    threshMap = cv2.threshold(saliencyMap.astype("uint8"), 0, 255,
                              cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    Image.fromarray((saliencyMap * 255).astype("uint8")).show()
    #cv2.imshow("tttt", threshMap)









