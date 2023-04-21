import cv2 as cv 
import numpy as np 
import requests

import cmd_canny2image 
import clip_interrogator_app

def url_to_cv2(url):
    # 发送HTTP请求并获取响应
    response = requests.get(url)
    
    # 将响应内容转换为OpenCV图像
    img_array = np.array(bytearray(response.content), dtype=np.uint8)
    img = cv.imdecode(img_array, cv.IMREAD_COLOR)
    
    # 返回OpenCV图像
    return img

if '__main__' == __name__:
    img_style=url_to_cv2('https://drive.studyinglover.com/api/raw/?path=/photos/blog/background/1679396986874.png')
    
    describe=clip_interrogator_app.app(img_style)
    
    img=url_to_cv2('https://drive.studyinglover.com/api/raw/?path=/photos/blog/background/1679397008541.png')
    
    out=cmd_canny2image.process(img,describe)

    cv.imwrite('out.png',out[1])