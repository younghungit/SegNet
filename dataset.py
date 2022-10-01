import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from glob import glob
import torch

class dataset(Dataset):
    """dataset
        -- data_dir
            |-- train
                |-- x
                    |-- 000.png
                    |-- ...
                    |-- 999.png
                |-- x_masked
                    |-- 000.png
                    |-- ...
                    |-- 999.png
    """
    def __init__(self, mode, data_dir, transform=None):
        # data_dir: (directory) Folder directory of datasets [ex) '.\comics_example']
        assert mode in ['train', 'test']
        self.transform = transform
        self.input_files, self.tar_files = self.build_file_list(mode, data_dir)
        # print(self.input_files)
        # print(self.tar_files)

    def build_file_list(self, mode, dir):
        if mode == 'train':
            # 데이터 합친 후 체크
            input_folder = os.path.join(dir, 'train', 'x')
            tar_folder = os.path.join(dir, 'train', 'x_masked')
            input_files = glob(os.path.join(input_folder,'*.png'))
            tar_files = glob(os.path.join(tar_folder,'*.png'))

        else:
            input_folder = os.path.join(dir, 'test', 'x')
            tar_folder = os.path.join(dir, 'test', 'x_masked')
            input_files = glob(os.path.join(input_folder, '*.png'))
            tar_files = glob(os.path.join(tar_folder, '*.png'))
            # assert NotImplementedError

        return input_files, tar_files

    def __getitem__(self, idx):
        toTensor = transforms.ToTensor()
        toResize = transforms.Resize((384, 384))
        input_file = self.input_files[idx]  # ./dataset/test/x/1001.png
        masked_file = input_file.replace('x', 'x_masked')
        fn = input_file.split('\\')[-1][:-4]  #1001

        # P4.3. Convert it to torch.LongTensor with shape ().
        # input_image = Image.open(input_file).convert('RGB')
        # masked_image = Image.open(masked_file).convert('RGB')
        input_image = Image.open(input_file).convert('L')
        masked_image = Image.open(masked_file).convert('L')
        input_image = toResize(toTensor(input_image))
        masked_image = toResize(toTensor(masked_image))
        # print(input_image.size())
        # print(masked_image.size())
        # 위쪽 부분을 잘라내는 전처리 코드 작성 후 다시 train.

        # print(input_image.size(), masked_image.size(), input_file, masked_file)
        # This is the trick (concatenate the images):
        # input image size 2x3x384x384 1x3x384x384
        both_images = torch.cat((input_image.unsqueeze(0), masked_image.unsqueeze(0)), 0)

        if self.transform is not None:
            transformed_images = self.transform(both_images)
        else:
            transformed_images = both_images
        # if self.transform is not None:
        #     masked_image = self.transform(masked_image)

        # Get the transformed images:
        input_image = transformed_images[0]
        masked_image = transformed_images[1]
        return input_image, masked_image, fn

    def __len__(self):
        return len(self.input_files)
