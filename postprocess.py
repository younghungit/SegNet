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
from PIL import ImageChops
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
    img_x_masked = Image.open(dir_masked)

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
        text = check_text(text)
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

def merge_box(box1, box2):
    x1 = min(box1[0], box2[0])
    y1 = min(box1[1], box2[1])
    x2 = max(box1[2], box2[2])
    y2 = max(box1[3], box2[3])
    return [x1, y1, x2, y2]


def merge_boxes(boxes):
    n = len(boxes)
    mbox = boxes[0]
    for i in range(n):
        mbox = merge_box(mbox, boxes[i])
    return mbox


def point_distance(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def box_distance(box1, box2):
    retval = point_distance(((box1[0] + box1[2]) / 2, (box1[1] + box1[3]) / 2),
                            ((box2[0] + box2[2]) / 2, (box2[1] + box2[3]) / 2))
    retval -= point_distance((box1[0], box1[1]), (box1[2], box1[3])) / 2
    retval -= point_distance((box2[0], box2[1]), (box2[2], box2[3])) / 2

    return retval


def clustering(boxes, threshold=50):
    boxes = [tuple(b) for b in boxes]
    clusters = []
    n = len(boxes)
    for i in range(n):
        cluster = {boxes[i]}
        for x, c in enumerate(clusters):
            if boxes[i] in c:
                cluster = c
                idx = x
        if cluster in clusters:
            del clusters[idx]
        for j in range(i + 1, n):
            if box_distance(boxes[i], boxes[j]) < threshold: cluster.add(boxes[j])
        clusters.append(cluster)

    retval = [merge_boxes(list(c)) for c in clusters]
    return retval


def main():
#     ###### network test 과정에서 추출한 이미지를 가져오는 function 만들기 ######
#     # result_img = Image.fromarray(result_img)
#     # # invert image
#     # result_img = ImageOps.invert(result_img)
#     # x_masked = gt_img - result_img

#     # img = get_txtimg(gt_img, x_masked)
#     #####################################################################
    gt_dir = 'D:/MetaCodeProject/segnet_220823/result_img/10_gt_img'
    cropped_dir = 'D:/MetaCodeProject/segnet_220823/result_img/cropped_img'
    inv_test_dir = sorted(os.listdir('D:/MetaCodeProject/segnet_220823/result_img/10_inv_img'))
    inv_dir = 'D:/MetaCodeProject/segnet_220823/result_img/10_inv_img'
    result_dir = 'D:/MetaCodeProject/segnet_220823/result_img/10_result_img'
    translated_dir = 'D:/MetaCodeProject/segnet_220823/result_img/translated_img'
    masked_dir = 'D:/MetaCodeProject/segnet_220823/result_img/masked_img'
    
    # gt_dir = 'D:/MetaCodeProject/segnet_220823/result_img/result_gt_img'
    # cropped_dir = 'D:/MetaCodeProject/segnet_220823/result_img/cropped_img'
    # inv_test_dir = sorted(os.listdir('D:/MetaCodeProject/segnet_220823/result_img/inv_test_img'))
    # inv_dir = 'D:/MetaCodeProject/segnet_220823/result_img/inv_test_img'
    # translated_dir = 'D:/MetaCodeProject/segnet_220823/result_img/translated_img'
    for i,inv_list in enumerate(inv_test_dir):
        # inv_list = '
        inv_img = os.path.join(inv_dir,inv_list) # './result_img/inv_test_img/inv_predicted_1001.png'
        gt_img = os.path.join(gt_dir, inv_list)
         # './result_img/result_gt_img/predicted_masked_1001.png'
        # print(inv_img)
        # print(gt_img)
        img = Image.open(inv_img).convert('L')
        img_gt = Image.open(gt_img).convert('L')
        # img_gt = img_gt.resize((384,384))
        # print(img)
        # print(img_gt)
        # width, height = img.size
        # draw = ImageDraw.Draw(img)
        # print()


        # cluster 생성
        txt, d = img2txt(img_gt)
        # print(check_text(txt))
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



        # draw.rectangle([(df_conf_positive['left'].min(),df_conf_positive['top'].min()),
        # (df_conf_positive['left_width'].max(),df_conf_positive['top'].max()+df_conf_positive['height'].max())])

        ######### clustering code 직쩝 짜보기 #########
        # clusters = [df_conf_positive['left'].min(),df_conf_positive['top'].min(),
        # df_conf_positive['left_width'].max(),df_conf_positive['top'].max()+df_conf_positive['height'].max()]
        df_conf_positive['text'] = df_conf_positive['text'].str.strip()
        df_conf_positive['text'].replace('',np.NaN,inplace = True)
        df_conf_positive.dropna(inplace = True)
        # print('.' + df_conf_positive['text'].iloc[4] + '.')
        # print(df_conf_positive['block_num'].unique())
        clusters = []
        for i in df_conf_positive['block_num'].unique():
            cluster = (df_conf_positive[df_conf_positive['block_num']==i]['left'].min(), df_conf_positive[df_conf_positive['block_num']==i]['top'].min(),
            df_conf_positive[df_conf_positive['block_num']==i]['left_width'].max(), 
            df_conf_positive[df_conf_positive['block_num']==i]['top'].max() + df_conf_positive[df_conf_positive['block_num']==i]['height'].max())

            clusters.append(cluster)

        
        # print(clusters)
        # assert  1==2
        #############################################
        # img.save('tmp_gray_box.png')

        
        croppedImageList = []
        text_locations = []
        translated_texts = []
        # image = Image.open("tmp.png")
        for i, cluster in enumerate(clusters): 
            croppedImage = img_gt.crop(cluster)
            croppedImage.save(os.path.join(cropped_dir,  inv_list[:-4] +'_{}.png'.format(i)))
            text_locations.append(cluster)
            text, _ = img2txt(croppedImage, lang='eng')
            text = check_text(text)
            text = text.replace('\n','')
            print(text)
            trans = Translator()
            translated_text = translate(trans, text, 'en', 'ko')
            if translated_text == None:
                translated_texts.append('')
                continue
            translated_texts.append(translated_text)
        print(text_locations)
        print(translated_texts)


        masked_img = os.path.join(masked_dir,inv_list)
        masked = Image.open(masked_img)
        
        # gt_img = Image.open(gt_img).resize((384,384))
        # masked =Image.fromarray(np.asarray(gt_img) - np.asarray(result_img))
        # masked = ImageChops.subtract(gt_img,result_img)

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

        masked.save(os.path.join(translated_dir, inv_list))

    # croppedImageList = sorted(os.listdir(cropped_dir))
    # trans = Translator()
    # translated_texts = []
    # for images in croppedImageList:
    #     cropped_img = Image.open(os.path.join(cropped_dir,images)).convert('L')
    #     text, _ = img2txt(cropped_img, lang='eng')
    #     text = check_text(text)
    #     translated_text = translate(trans, text, 'en', 'ko')
    #     if translated_text == None:
    #         translated_texts.append('')
    #         continue
    #     translated_texts.append(translated_text)
    #     # print(translated_text)
    #     # print('\n')

    #     # masked = Image.fromarray(np.asarray(Image.open(gt_img))-np.asarray(Image.open(inv_img))) # ground_truth - model_passed_img 롭 바꾸어주고


    
    # for inv_list in inv_test_dir:
    #     # result_img = os.path.join(result_dir,inv_list) # './result_img/inv_test_img/inv_predicted_1001.png'
    #     # gt_img = os.path.join(gt_dir, inv_list)
    #     # inv_img = os.path(inv_dir,inv_list)
    #     # result_img = Image.open(result_img)
    #     # inv_fin_img = Image.open(inv_img)

    #     masked_img = os.path.join(masked_dir,inv_list)
    #     masked = Image.open(masked_img)
        
    #     # gt_img = Image.open(gt_img).resize((384,384))
    #     # masked =Image.fromarray(np.asarray(gt_img) - np.asarray(result_img))
    #     # masked = ImageChops.subtract(gt_img,result_img)

    #     fnt = "D:/MetaCodeProject/segnet_220823/font/NanumGothicBold.ttf"
    #     size = 20 #min(int(height / 12), 24)
    #     font = ImageFont.truetype(fnt, size)
    #     draw = ImageDraw.Draw(masked)
    #     for i, text in enumerate(translated_texts):
    #         location = text_locations[i]
    #         width = int((location[2] - location[0]) / (0.45 * size))

    #         for j in range(len(text) // width + 1):
    #             sub_text = text[width * j:width * (j + 1)]
    #             draw.text((location[0], location[1] + size * j), sub_text, 0, font=font)

    #             masked.save(os.path.join(translated_dir, inv_list))

if __name__ == "__main__":
    # main('D:/MetaCodeProject/segnet_220823/dataset/test/x/1001.png','D:/MetaCodeProject/segnet_220823/dataset/test/x_masked/1001.png')
    # main('D:/MetaCodeProject/segnet_220823/dataset/test/x/1031.png','D:/MetaCodeProject/segnet_220823/dataset/test/x_masked/1031.png')
    # get_passedimg('D:/MetaCodeProject/segnet_220823/result_img/result_img')
    main()