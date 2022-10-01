from PIL import Image
from PIL import ImageOps
import numpy as np
import pytesseract


def img2txt(img):
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    txt = pytesseract.image_to_string(img, lang='eng')
    d = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)

    return txt,d

def get_txtimg(dir, dir_masked):
    img_x = Image.open(dir)
    img_x_masked = Image.open(dir_masked)

    # Get the images as ndarray
    img_x = np.asarray(img_x.convert('L'))
    img_x_masked = np.asarray(img_x_masked.convert('L'))

    # Subtract image x from image x maksed
    img_diff = img_x_masked - img_x
    # print(img_diff.shape)
    h,w = img_diff.shape
    # img_diff = img_diff[:int(h/3),int(w/2):-int(w/8)]

    # Construct a new Image from the resultant buffer
    img_diff = Image.fromarray(img_diff)
    img_diff.save('tmp_inv.png')

    # invert image
    img_diff_inv = ImageOps.invert(img_diff)
    img_diff_inv.save('tmp.png')
    # assert 1==2
    return img_diff_inv

def main(dir, dir_masked):
    # txtimg = get_txtimg(dir, dir_masked)
    # txt, d = img2txt(txtimg)
    txt, d = img2txt(Image.open('tmp_crop.png'))
    print(txt)
    print(d)

if __name__ == "__main__":
    x = './comics_example/train/x/1.png'
    x_masked = './comics_example/train/x_masked/1.png'
    main(x, x_masked)