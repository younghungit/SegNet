import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from model2 import SegNet
from torch.optim.lr_scheduler import MultiStepLR
from dataset import dataset
from random import randint
# from test import testcode

# use_gpu = torch.cuda.is_available()
# ngpu = torch.cuda.device_count()
# device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='./dataset')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--log_dir', type=str, default='./logs')
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--batch_momentum', type=int, default=0.1)
    parser.add_argument('--epochs', type=int, default=5000)
    parser.add_argument('--learning_rate', type=int, default=0.0001)
    parser.add_argument('--beta1', type=int, default=0.9)
    parser.add_argument('--beta2', type=int, default=0.999)
    parser.add_argument('--weight_decay', type=int, default=0.0001)
    parser.add_argument('--eval_step', type=int, default=500)
    parser.add_argument('--mode', type=str, default='train', help='train, test')
    parser.add_argument('--gpu', type=str, default='7')
    args = parser.parse_args()
    return args

def train_net(args):
    # degree = randint(-180, 180)
    
    transforms = torchvision.transforms.Compose([
        # torchvision.transforms.Resize((256, 256)),
        torchvision.transforms.RandomRotation((-180, 180)),
        torchvision.transforms.RandomHorizontalFlip(p=0.5),
        torchvision.transforms.RandomVerticalFlip(p=0.5),
        # torchvision.transforms.ToTensor()
    ])

    train_data = dataset(
        mode='train',
        data_dir=args.data,
        transform=transforms
        )

    test_data = dataset(
        mode='test',
        data_dir=args.data,
        transform=None
    )

    # 주석 1.
    # 원래는 ckpt_dict['model'] 이라는 키값을 가져와서 썼는데
    # 그렇게하면 코드 돌리면서 저장 할 때 
    # torch.save({
            # 'epoch': epoch,
            # 'model_state_dict': model.state_dict(),
            # 'optimizer_state_dict': optimizer.state_dict(),
            # 'loss': loss,
            # ...
            # }, PATH)
    # 저 코드가 없는데 음,, checkpoint 불러올 때 키값자체가 존재하지 않는 것 같습니다,,
    # 저 torch.save코드를 eval 코드 그러니까 with torch.no_grad()문 바깥쪽에 넣어야하나요?
    # 144번째 줄에 넣는게 맞을까요,,?
    segnet = SegNet()
    segnet.cuda()
    optimizer = torch.optim.Adam(segnet.parameters(), lr=args.learning_rate, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay)
    scheduler = MultiStepLR(optimizer, milestones=LR_milestones, gamma=0.5)
    if args.checkpoint is not None:
        ckpt_dict = torch.load(args.checkpoint)
        segnet.load_state_dict(ckpt_dict['model'])
        optimizer.load_state_dict(ckpt_dict['optimizer'])
        step = ckpt_dict['iteration']
        epoch = ckpt_dict['epoch']
        print('checkpoint loaded!')
    else:
        step, epoch = 0, 0

    train(segnet, train_data, test_data, args.epochs, step, epoch, optimizer, scheduler, args)

def train(model, train_data, test_data, epochs, step, epoch, optimizer, scheduler, args):
    print('training started')
    writer = SummaryWriter(args.log_dir)

    train_dataloader = DataLoader(train_data,
                            shuffle=True,
                            batch_size=args.batch_size,
                            drop_last=True)

    test_dataloader = DataLoader(test_data,
                            shuffle=False, # 테스트는 항상 같은 것을 출력해야
                            batch_size=args.batch_size,
                            drop_last=True)

    criterion = nn.MSELoss(reduction='mean')

    model.train()
    for _ in range(epochs):  # epoch : 1000 dataset 1 training
        # print( 'Epoch {}/{}'.format(epoch, epochs -1))
        for i, batch in enumerate(train_dataloader):
            step += 1
            # 데이터 획득, 래핑
            input_img, maksed_img, _ = batch
            input_img = input_img.cuda()
            maksed_img = maksed_img.cuda()
            predicted = model(input_img)  # predicted = 추출한 text
            # print(input_img.size(), predicted.size())

            loss = criterion(predicted, maksed_img-input_img) * 1000.0  # predicts extracted text in white, all others in black
            # print(loss)
            optimizer.zero_grad()  # 파라미터 기울기 최적화
            loss.backward()
            optimizer.step()
            scheduler.step


            # 주석 2. 여기서 Input 값은 Tensor (3,12,224,224) Input_img 는 CHW 라는 형식의 값으로
            # 에러가 뜨는데 이 부분 잘 해결이 안되서 그냥 날렸습니다.
            # 아마 dataset.py에서 RGB로 갖고오면서 그런 것이라 예상이 되는데,,,
            # 오류메시지 갖다가 구글에 넣어도 잘 안나오고,,(사실 주석에 쓴 것들에 어느정도 들어맞는 그러한 답변이 딱히 없습니다)
            # print(predicted.size())
            # print(predicted[0].size())
            if step % 10 == 0:
                print('######### Epoch {}, Step {}, loss: {:.4f} #########'.format(epoch, step, loss.item()))
                writer.add_scalar('train/loss', loss.cpu().item(), step)
                writer.add_image('train/pred_img/1', predicted[0].cpu().detach().numpy(), step)
                writer.add_image('train/gt_img/1', (maksed_img - input_img)[0].cpu().detach().numpy(), step)
                writer.add_image('train/pred_img/2', predicted[1].cpu().detach().numpy(), step)
                writer.add_image('train/gt_img/2', (maksed_img - input_img)[1].cpu().detach().numpy(), step)
                writer.add_image('train/pred_img/3', predicted[2].cpu().detach().numpy(), step)
                writer.add_image('train/gt_img/3', (maksed_img - input_img)[2].cpu().detach().numpy(), step)
                
                # Loss 값을 따로 추가 시켜줘야하는데 여기서 계속 오류나서 인터넷 검색중입니다

            #데이터 수가 작기 때문에 한 에폭당 평가
        with torch.no_grad():       #evaluation 할 때는 gradiant update x
            model.eval()
            print('begin evaluation')
            test_loss = []
            for j, batch_test in enumerate(test_dataloader):
                test_input_img, test_maksed_img, _ = batch_test
                test_input_img = test_input_img.cuda()
                test_maksed_img = test_maksed_img.cuda()
                output = model(test_input_img)
                loss = criterion(output, test_maksed_img - test_input_img)
                test_loss.append(loss.item())

            print('######### Average Evaluation Loss : {} #########'.format(np.mean(test_loss)*1000))
            writer.add_scalar('test/loss', np.mean(test_loss), epoch)
            writer.add_image('test/pred_img/1', output[0].cpu().detach().numpy(), epoch)
            writer.add_image('test/gt_img/1', (test_maksed_img - test_input_img)[0].cpu().detach().numpy(), epoch)
            writer.add_image('test/pred_img/2', output[1].cpu().detach().numpy(), epoch)
            writer.add_image('test/gt_img/2', (test_maksed_img - test_input_img)[1].cpu().detach().numpy(), epoch)
            writer.add_image('test/pred_img/3', output[2].cpu().detach().numpy(), epoch)
            writer.add_image('test/gt_img/3', (test_maksed_img - test_input_img)[2].cpu().detach().numpy(), epoch)

            # testcode(test_dataloader, model)
            # 주석 3.
            # 분명히 print를 해서 넣은 것 같은데 자꾸 안떠서 이것저것 했는데도 잘 안됐습니다
            # 주 오류는 test_loss 인데
            # test_loss를 for 문에서 하나 씩 더해줬고
            # 이 후 for 문 밖에서 test_dataloader를 활용해서 나눠서 
            # 평균 값을 도출 하는데 
            # 이 과정에서 for문 밖에 test_loss를 정의 하는 과정에서
            # 이미 for 문 안쪽에 local_varaible로 정의 되었다는 오류가 지속적으로 뜹니다,,
            # 그래서 다른 변수에 저장해서 해도 print 자체가 안되구요,,

        model.train()  # evaluation stop, training restart

        if not os.path.exists(args.checkpoint_dir):
            os.mkdir(args.checkpoint_dir)
        torch.save({'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'iteration': step,
                    'epoch': epoch},
                   os.path.join(args.checkpoint_dir, 'ckpt_{:04d}.pth'.format(epoch)))
        epoch += 1

    writer.close()
    print('training ended')

# def testcode(test_dataloader, model):
#     for j, batch_test in enumerate(test_dataloader):
#         test_input_img, test_maksed_img = batch_test
#         test_input_img = test_input_img.cuda()
#         test_maksed_img = test_maksed_img.cuda()
#         output = model(test_input_img)
#         criterion = nn.MSELoss(reduction='sum')
#         loss = criterion(output, test_maksed_img - test_input_img)
#         test_loss += loss.item()

#     test_loss /= len(test_dataloader.dataset)
#     print('Average Loss : {}'.format(test_loss))
#     print('Test_numpy Image : {}'.format(test_loss.to_numpy()) )



if __name__ == "__main__":
    torch.cuda.empty_cache()
    args = parse_args()
    train_net(args)
    # 주석 4
    # 어떻게 어떻게 트레이닝이 되면서 각 weight가 저장되는 것 까진 봤는데
    # 막상 tensorboard 열어서 실시간으로 확인해보니
    # loss가 줄지 않는 것을 보고 음,, 뻘짓했구나 싶어서 
    # 나머지 주석 해결을 해야 할 것 같습니다,,