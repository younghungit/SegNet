import os
import argparse
import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np
from glob import glob
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from model2 import SegNet
from torch.optim.lr_scheduler import MultiStepLR
from dataset import dataset
from random import randint
from PIL import Image



def predict():
    bn = 6
    model = SegNet()
    checkpoint = torch.load('D:\MetaCodeProject/segnet_220823/checkpoints/lr_0.00001_new/ckpt_0049.pth')
    model.cuda()
    model.load_state_dict(checkpoint['model'])
    model.eval()
    data_dir ='./dataset'

    test_data = dataset(
        mode='test',
        data_dir=data_dir,
        transform=None
    )

    test_dataloader = DataLoader(test_data,
                        shuffle=False, # 테스트는 항상 같은 것을 출력해야
                        batch_size=bn,
                        drop_last=True)
    
    for i, batch in enumerate(test_dataloader):
        # 데이터 획득, 래핑
        input_img, _, filename = batch
        input_img = input_img.cuda()
        with torch.no_grad():
            predicted = model(input_img)
        
        # save results
        for j in range(bn):

            # in_img = input_img[j].squeeze(0).contiguous().cpu().detach().numpy()
            in_img = input_img[j].squeeze(0).contiguous().cpu().detach()

            # gt_img = (test_maksed_img - test_input_img)[j].cpu().detach().numpy()
            # result_img = predicted[j].squeeze(0).contiguous().cpu().detach().numpy()
            result_img = predicted[j].squeeze(0).contiguous().cpu().detach()
            
            result_masked_img = in_img- result_img
            # in_img = Image.fromarray(in_img).convert('L')
            # result_img = Image.fromarray(result_img).convert('L')
            # result_masked_img = Image.fromarray(result_masked_img).convert('L')
            result_img = torchvision.utils.save_image(result_img,'./result_img/result_img/predicted_{}.png'.format(filename[j]))
            result_masked_img = torchvision.utils.save_image(result_masked_img,'./result_img/result_maksed_img/predicted_masked_{}.png'.format(filename[j]))
            # print('./result_img/result_img/predicted_{}.png'.format(filename[j]))
            # result_img.save('./result_img/result_img/predicted_{}.png'.format(filename[j]))
            # in_img.save('input/input_{}'.format(j))
            # result_masked_img.save('./result_img/result_maksed_img/predicted_masked_{}.png'.format(filename[j]))
            # gt_img.save('gt/gt_{}'.format(j))


if __name__ == "__main__":
    predict()