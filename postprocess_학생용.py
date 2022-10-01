import os
import sys
import glob
import cv2
import requests
import getopt
import math
import numpy as np
from PIL import Image
from PIL import ImageOps
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
import pandas as pd
import pytesseract
from googletrans import Translator
from papago import Translator as Translator_papago


def img2txt(img, lang='eng'):
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    txt = pytesseract.image_to_string(img, lang=lang)
    d = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)

    return txt,d

def get_passedimg(dir):
    list_dir = os.listdir(dir)
    for list in list_dir:
        test_img = Image.open(dir+'/'+list)
    # invert image
        test_img_inv = ImageOps.invert(test_img)
        test_img_inv.save('D:/MetaCodeProject/segnet_220823/result_img/inv_test_img/inv_' + list)
def get_txtimg(dir, dir_masked):
    img_x = Image.open(dir)
    # img_x_masked = Image.open(dir_masked)

    # Get the images as ndarray
    img_x = np.asarray(img_x.convert('L'))
    img_x_masked = np.asarray(img_x_masked.convert('L'))

    # Subtract image x from image x maksed
    # img_diff = img_x_masked - img_x
    # print(img_diff.shape)
    # h,w = img_diff.shape
    # img_diff = img_diff[:int(h/3),int(w/2):-int(w/8)]

    # Construct a new Image from the resultant buffer
    # img_diff = Image.fromarray(img_diff)
    # img_diff.save('tmp_inv.png')

    # invert image
    # img_diff_inv = ImageOps.invert(img_diff)
    # img_diff_inv.save('tmp.png')
    # assert 1==2
    # img


    # return img_diff_inv


def translate(trans, text, source, target):
    if text == '': return
    if source == 'en':
        text = check_text(text.replace('\n', ' ').lower())
    elif source == 'ko':
        text = text.replace('\n', ' ')
    # print(text)
    try:
        translated_text = translate_papago(text, source=source, target=target)
        # print('finish translating with papago')
    except:
        translated_text = translate_google(trans, text, dest=target)
        # print('finish translating with google')

    return translated_text


def translate_papago(text, source='en', target='ko', honorific='true'):
    request_url = "https://openapi.naver.com/v1/papago/n2mt"
    headers = {"X-Naver-Client-Id": "NUTz_cMvRLS2L1aT84pJ", "X-Naver-Client-Secret": "uXcsa9YJ3c"}
    params = {"honorific": honorific, "source": source, "target": target, "text": text}
    response = requests.post(request_url, headers=headers, data=params)
    result = response.json()
    return result['message']['result']['translatedText']


def translate_google(trans, text, dest='ko'):
    result = trans.translate(text, dest=dest)
    return result.text


def check_text(text):
    # for char in text:
    #     if char not in 'abcdefghijklmnopqrstuvwxyz1234567890,.?!()[]{}\'- ':
    #         text = text.replace(char, '')
    text = text.replace('1AM', 'I AM')
    text = text.replace('yoo', 'you')
    text = text.replace('WOU','YOU')
    text = text.replace('is)','IN')
    text= text.replace('FER','FOR')
    text = text.replace('"PVE','I\'VE')
    text = text.replace('1',"I")
    ### make more replacement
    return text


def main(gt_img, x_masked):
    ###### network test 과정에서 추출한 이미지를 가져오는 function 만들기 ######
    # result_img = Image.fromarray(result_img)
    # # invert image
    # result_img = ImageOps.invert(result_img)
    # x_masked = gt_img - result_img

    # img = get_txtimg(gt_img, x_masked)
    #####################################################################
    gt_dir = 'D:/MetaCodeProject/segnet_220823/result_img/result_gt_img'
    inv_test_dir = sorted(os.list_dir('D:/MetaCodeProject/segnet_220823/result_img/inv_test_img'))
    for i,inv_list in enumerate(inv_test_dir):
        # inv_list = '
        inv_img = os.path.join(inv_test_dir,inv_list) # './result_img/inv_test_img/inv_predicted_1001.png'
        gt_img = os.path.join(gt_dir, inv_img.split('/')[-1].replace('inv_predicted','predicted_masked'))
         # './result_img/result_gt_img/predicted_masked_1001.png'

        img = Image.open(inv_img)
        image = img

        width, height = img.size
        draw = ImageDraw.Draw(img)
        txt, d = img2txt(img)
        print(check_text(txt))
        # print(d)
        n_boxes = len(d['level'])
        confidences = d['conf']
        width = d['width']
        height = d['height']
        boxes = []
        df = pd.DataFrame(d)
        
        df_conf_positive = df[df['conf']>=1]
        df_conf_positive = check_text(df_conf_positive)
        # print(df_conf_positive)
        # print(df_conf_positive['left'].mean())
        # print(df_conf_positive[df_conf_positive['left']<df_conf_positive['left'].mean()-df_conf_positive['width'].mean()]['text'])
        # if int(confidences) >= 0:
        df_conf_positive['left_width'] = df_conf_positive['left'] + df_conf_positive['width']
        # for i in range(n_boxes):
        #     if int(float(confidences[i])) >= 0:
        #         (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
        #         # print(area(w, h))
        #         if w*h < 100 or w*h > 6000: continue
        #         boxes.append([x, y, x + w, y + h])
        #         draw.rectangle([(x, y), (x + w, y + h)])
        draw.rectangle([(df_conf_positive['left'].min(),df_conf_positive['top'].min()),
        (df_conf_positive['left_width'].max(),df_conf_positive['top'].max()+df_conf_positive['height'].max())])

        ######### clustering code 직쩝 짜보기 #########
        clusters = [df_conf_positive['left'].min(),df_conf_positive['top'].min(),
        df_conf_positive['left_width'].max(),df_conf_positive['top'].max()+df_conf_positive['height'].max()]


        clusters = 
        #############################################
        # img.save('tmp_gray_box.png')

        alpha = 0
        croppedImageList = []
        text_locations = []
        # image = Image.open("tmp.png")
        for i, cluster in enumerate(clusters): 
            for j, num_culster in enumerate(cluster):
                croppedImage = image.crop((cluster[0] - alpha, cluster[1] - alpha, cluster[2] + alpha, cluster[3] + alpha))
                croppedImage.save("tmp_crop_{}.jpg".format(i))
                text_locations.append(cluster)
            croppedImage = image.crop(clusters)
            croppedImage.save("tmp_crop_.jpg")
            text_locations.append(clusters)

        croppedImageList = sorted(glob.glob("tmp_crop_*.jpg"))
        trans = Translator()
        translated_texts = []
        for image in croppedImageList:
            text, _ = img2txt(Image.open(image), lang='eng')
            text = check_text(text)
            # print(text)
            translated_text = translate(trans, text, 'en', 'ko')
            if translated_text == None:
                translated_texts.append('')
                continue
            translated_texts.append(translated_text)
            # print(translated_text)
            # print('\n')

        masked = Image.open(gt_img-img) # ground_truth - model_passed_img 롭 바꾸어주고

        fnt = "D:/MetaCodeProject/segnet_220823/font/NanumGothicBold.ttf"
        size = 20 #min(int(height / 12), 24)
        font = ImageFont.truetype(fnt, size)
        draw = ImageDraw.Draw(masked)

        for i, text in enumerate(translated_texts):
            location = text_locations[i]
            width = int((location[2] - location[0]) / (0.45 * size))

            for j in range(len(text) // width + 1):
                sub_text = text[width * j:width * (j + 1)]
                draw.text((location[0], location[1] + size * j), sub_text, 0, font=font)

        masked.save("translated.png")

if __name__ == "__main__":
    # main('D:/MetaCodeProject/segnet_220823/dataset/test/x/1001.png','D:/MetaCodeProject/segnet_220823/dataset/test/x_masked/1001.png')
    # main('D:/MetaCodeProject/segnet_220823/dataset/test/x/1031.png','D:/MetaCodeProject/segnet_220823/dataset/test/x_masked/1031.png')
    get_passedimg('D:/MetaCodeProject/segnet_220823/result_img/result_img')