from clip_interrogator import Config, Interrogator
import cv2 as cv
from PIL import Image

def app(numpy_img):
    img = cv.cvtColor(numpy_img,cv.COLOR_BGR2RGB)
    img = Image.fromarray(img)

    ci = Interrogator(Config(clip_model_name="ViT-L-14/openai"))

    describe=ci.interrogate(img)
    return describe

if '__main__'==__name__:
    ans=app(cv.imread('test.png'))
    print(ans)