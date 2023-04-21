from clip_interrogator import Config, Interrogator
import cv2 as cv
from PIL import Image

img=cv.imread('test.png')
img = cv.cvtColor(img,cv.COLOR_BGR2RGB)
img = Image.fromarray(img)

ci = Interrogator(Config(clip_model_name="ViT-L-14/openai"))

describe=ci.interrogate(img)
print('\n'+describe)