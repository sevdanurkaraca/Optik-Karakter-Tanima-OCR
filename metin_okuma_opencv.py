import cv2
import numpy as np
import pytesseract
from PIL import Image

#resim dosyalarının bulunduğu yerin yolunu tanımlıyoruz
src_path = "D:\\ocrProje\\metin_okuma_openCv"

#pythona tesseractı göstermek için
pytesseract.pytesseract.tesseract_cmd = 'C:\Program Files\Tesseract-OCR\tesseract'

def get_string(img_path):
    img=cv2.imread(img_path)
    height = np.size(img,0)
    width = np.size(img,1)
    res = cv2.resze(img,(5*width,5*height), interpolation= cv2.INTER_CUBIC)
    cv2.imwrite(src_path+"thres.png",res)

    result = pytesseract.image_to_string(Image.open(src_path+"thres.png"))
    return result
print("--- Baslangıç ---")
print(get_string(src_path+'metin_okuma_openCv'))
