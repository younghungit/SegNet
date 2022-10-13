from PIL import Image
import cv2
import os
from cv2 import imshow
from glob import glob

def image_convert_bw (file_name, save_path, comic_num):
  originalImage = cv2.imread(file_name,cv2.IMREAD_GRAYSCALE)
#   originalImage = cv2.imread(file_name)
#   grayImage = cv2.cvtColor(originalImage, cv2.COLOR_BGR2GRAY)
#   (thresh, blackAndWhiteImage) = cv2.threshold(grayImage, 127, 255, cv2.THRESH_BINARY)
#   (thresh, blackAndWhiteImage) = cv2.threshold(originalImage, 127, 255, cv2.THRESH_BINARY)    
  (thresh, blackAndWhiteImage) = cv2.threshold(originalImage, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

  fname = '{}.png'.format(comic_num)
  save_name = save_path + fname
  cv2.imwrite(save_name, blackAndWhiteImage)
  print('save file ' + save_name + '....')

file_names = glob("D:/MetaCodeProject/ImageDataset/Zip/022_mask/*")
comic_num = 377

if __name__ == '__main__':
  for file_name in file_names:
    image_convert_bw( file_name, 'D:/MetaCodeProject/ImageDataset/re_bw/x_masked/', comic_num)
    comic_num += 1



# img_dir = 'D:/MetaCodeProject/ImageDataset/022_grayscale/'
# img_list = os.listdir(img_dir)
# for item in img_list:
#     print(img_dir+item)
#     # img = Image.open(img_dir + item).convert('L')
#     originalImage = cv2.imread(img_dir + item)
#     grayImage = cv2.cvtColor(originalImage, cv2.COLOR_BGR2GRAY)
#     (thresh, blackAndWhiteImage) = cv2.threshold(grayImage, 127, 255, cv2.THRESH_BINARY)
#     blackAndWhiteImage.save('D:/MetaCodeProject/ImageDataset/re_bw/x/'+item)