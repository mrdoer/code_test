# -*- coding: utf-8 -*-
# @Author: Song Dejia
# @Date:   2018-11-09 10:06:59
# @Last Modified by:   Song Dejia
# @Last Modified time: 2018-11-23 17:26:44
import os
import os.path as osp
import random
import time
import sys; sys.path.append('../')
import torch
import torch.nn as nn
import numpy as np
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import argparse
from PIL import Image, ImageOps, ImageStat, ImageDraw
from data_loader import TrainDataLoader
from net import SiameseRPN
from torch.nn import init
# from shapely.geometry import Polygon

parser = argparse.ArgumentParser(description='PyTorch SiameseRPN Training')

parser.add_argument('--train_path', default='D:/vot-data/vot2014/', metavar='DIR',help='path to dataset')
#/home/ly/chz/ILSVRC2015/Data/VID/train/ILSVRC2015_VID_train_0000
parser.add_argument('--weight_dir', default='D:/Huizhi/model/dsrpn/', metavar='DIR',help='path to weight')

parser.add_argument('--checkpoint_path', default=None, help='resume')
#/home/ly/chz/weights-0690000.pth.tar
parser.add_argument('--max_epoches', default=10000, type=int, metavar='N', help='number of total epochs to run')

parser.add_argument('--max_batches', default=0, type=int, metavar='N', help='number of batch in one epoch')

parser.add_argument('--init_type',  default='xavier', type=str, metavar='INIT', help='init net')

parser.add_argument('--lr', default=0.001, type=float, metavar='LR', help='initial learning rate')

parser.add_argument('--momentum', default=0.9, type=float, metavar='momentum', help='momentum')

parser.add_argument('--weight_decay', '--wd', default=5e-5, type=float, metavar='W', help='weight decay (default: 1e-4)')

parser.add_argument('--debug', default=False, type=bool,  help='whether to debug')

def main():
    """ train dataloader """
    args = parser.parse_args()
    data_loader = TrainDataLoader(args.train_path, check = args.debug)
    if not os.path.exists(args.weight_dir):
        os.makedirs(args.weight_dir)

    """ compute max_batches """
    for root, dirs, files in os.walk(args.train_path):
        for dirname in dirs:
            dir_path = os.path.join(root, dirname)
            args.max_batches += len(os.listdir(dir_path))
    print('max_batches: {}'.format(args.max_batches))
    """ Model on gpu """
    model = SiameseRPN()
    model = model.cuda()
    cudnn.benchmark = True

    """ loss and optimizer """
    criterion = MultiBoxLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay = args.weight_decay)

    """ load weights """
    init_weights(model)
    if not args.checkpoint_path == None:
        assert os.path.isfile(args.checkpoint_path), '{} is not valid checkpoint_path'.format(args.checkpoint_path)
        try:
            checkpoint = torch.load(args.checkpoint_path)
            start = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
        except:
            start = 0
            init_weights(model)
    else:
        start = 0

    """ train phase """
    closses, rlosses, tlosses = AverageMeter(), AverageMeter(), AverageMeter()
    steps = 0
    #print('data_loader length: {}'.format(len(data_loader)))
    for epoch in range(start, args.max_epoches):
        cur_lr = adjust_learning_rate(args.lr, optimizer, epoch, gamma=0.1)
        index_list = range(data_loader.__len__())
        example_index = 0
        for example in range(args.max_batches):
            ret = data_loader.__get__(random.choice(index_list))
            template = ret['template_tensor'].cuda()
            detection= ret['detection_tensor'].cuda()
            pos_neg_diff = ret['pos_neg_diff_tensor'].cuda()
            cout, rout = model(template, detection)
            predictions, targets = (cout, rout), pos_neg_diff
            closs, rloss, loss, reg_pred, reg_target, pos_index, neg_index = criterion(predictions, targets)
            closs_ = closs.cpu().item()

            if np.isnan(closs_):
               sys.exit(0)

            closses.update(closs.cpu().item())
            rlosses.update(rloss.cpu().item())
            tlosses.update(loss.cpu().item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            steps += 1

            cout = cout.cpu().detach().numpy()
            score = 1/(1 + np.exp(cout[:,0]-cout[:,1]))

            # ++++++++++++ post process below just for debug ++++++++++++++++++++++++
            # ++++++++++++++++++++ v1.0 add penalty +++++++++++++++++++++++++++++++++
            if ret['pos_anchors'] is not None:
                penalty_k = 0.055
                tx, ty, tw, th = ret['template_target_xywh'].copy()
                tw *= ret['template_cropprd_resized_ratio']
                th *= ret['template_cropprd_resized_ratio']

                anchors = ret['anchors'].copy()
                w = anchors[:,2] * np.exp(reg_pred[:, 2].cpu().detach().numpy())
                h = anchors[:,3] * np.exp(reg_pred[:, 3].cpu().detach().numpy())

                eps = 1e-2
                change_w = np.maximum(w/(tw+eps), tw/(w+eps))
                change_h = np.maximum(h/(th+eps), th/(h+eps))
                penalty = np.exp(-(change_w + change_h - 1) * penalty_k)
                pscore = score * penalty
            else:
                pscore = score

            # +++++++++++++++++++ v1.0 add window default cosine ++++++++++++++++++++++
            score_size = 17
            window_influence = 0.42
            window = (np.outer(np.hanning(score_size), np.hanning(score_size)).reshape(17,17,1) + np.zeros((1, 1, 5))).reshape(-1)
            pscore = pscore * (1 - window_influence) + window * window_influence
            score_old = score
            score = pscore #from 0.2 - 0.7

            # ++++++++++++++++++++ debug for class ++++++++++++++++++++++++++++++++++++
            if example_index%1000 == 0:
                print(score[pos_index])  # this should tend to be 1
                print(score[neg_index])  # this should tend to be 0


            # ++++++++++++++++++++ debug for reg ++++++++++++++++++++++++++++++++++++++
            tmp_dir = '/home/ly/chz/Siamese-RPN-pytorch/code_v1.0/tmp/visualization/7_check_train_phase_debug_pos_anchors'
            if not os.path.exists(tmp_dir):
                os.makedirs(tmp_dir)
            detection = ret['detection_cropped_resized'].copy()
            draw = ImageDraw.Draw(detection)
            pos_anchors = ret['pos_anchors'].copy() if ret['pos_anchors'] is not None else None

            if pos_anchors is not None:
                # draw pos anchors
                x = pos_anchors[:, 0]
                y = pos_anchors[:, 1]
                w = pos_anchors[:, 2]
                h = pos_anchors[:, 3]
                x1s, y1s, x2s, y2s = x - w//2, y - h//2, x + w//2, y + h//2
                for i in range(16):
                    x1, y1, x2, y2 = x1s[i], y1s[i], x2s[i], y2s[i]
                    draw.line([(x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)], width=1, fill='white') # pos anchor

                # pos anchor transform to red box after prediction
                x = pos_anchors[:,0] + pos_anchors[:, 2] * reg_pred[pos_index, 0].cpu().detach().numpy()
                y = pos_anchors[:,1] + pos_anchors[:, 3] * reg_pred[pos_index, 1].cpu().detach().numpy()
                w = pos_anchors[:,2] * np.exp(reg_pred[pos_index, 2].cpu().detach().numpy())
                h = pos_anchors[:,3] * np.exp(reg_pred[pos_index, 3].cpu().detach().numpy())
                x1s, y1s, x2s, y2s = x - w//2, y - h//2, x + w//2, y + h//2
                for i in range(16):
                    x1, y1, x2, y2 = x1s[i], y1s[i], x2s[i], y2s[i]
                    draw.line([(x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)], width=1, fill='red')  # predict(white -> red)

                # pos anchor should be transformed to green gt box, if red and green is same, it is overfitting
                x = pos_anchors[:,0] + pos_anchors[:, 2] * reg_target[pos_index, 0].cpu().detach().numpy()
                y = pos_anchors[:,1] + pos_anchors[:, 3] * reg_target[pos_index, 1].cpu().detach().numpy()
                w = pos_anchors[:,2] * np.exp(reg_target[pos_index, 2].cpu().detach().numpy())
                h = pos_anchors[:,3] * np.exp(reg_target[pos_index, 3].cpu().detach().numpy())
                x1s, y1s, x2s, y2s = x - w//2, y-h//2, x + w//2, y + h//2
                for i in range(16):
                    x1, y1, x2, y2 = x1s[i], y1s[i], x2s[i], y2s[i]
                    draw.line([(x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)], width=1, fill='green') # gt  (white -> green)
                x1, y1, x3, y3 = x1s[0], y1s[0], x2s[0], y2s[0]
            else:
                x1, y1, x3, y3 = 0, 0, 0, 0
            # top1 proposal after nms (white)
            if example_index%1000 == 0:
                save_path = osp.join(tmp_dir, 'epoch_{:010d}_{:010d}_anchor_pred.jpg'.format(epoch, example))
                detection.save(save_path)
            example_index = example_index+1

            # +++++++++++++++++++ v1.0 restore ++++++++++++++++++++++++++++++++++++++++
            ratio = ret['detection_cropped_resized_ratio']
            detection_cropped = ret['detection_cropped'].copy()
            detection_cropped_resized = ret['detection_cropped_resized'].copy()
            original = Image.open(ret['detection_img_path'])
            x_, y_ = ret['detection_tlcords_of_original_image']
            draw = ImageDraw.Draw(original)
            w, h = original.size
            """ un resized """
            x1, y1, x3, y3 = x1/ratio, y1/ratio, y3/ratio, y3/ratio

            """ un cropped """
            x1 = np.clip(x_ + x1, 0, w-1).astype(np.int32) # uncropped #target_of_original_img
            y1 = np.clip(y_ + y1, 0, h-1).astype(np.int32)
            x3 = np.clip(x_ + x3, 0, w-1).astype(np.int32)
            y3 = np.clip(y_ + y3, 0, h-1).astype(np.int32)

            draw.line([(x1, y1), (x3, y1), (x3, y3), (x1, y3), (x1, y1)], width=3, fill='yellow')
            #save_path = osp.join(tmp_dir, 'epoch_{:010d}_{:010d}_restore.jpg'.format(epoch, example))
            #original.save(save_path)

            print("Epoch:{:04d}\texample:{:06d}/{:06d}({:.2f})%\tsteps:{:010d}\tlr:{:.7f}\tcloss:{:.4f}\trloss:{:.4f}\ttloss:{:.4f}".format(epoch, example+1, args.max_batches, 100*(example+1)/args.max_batches, steps, cur_lr, closses.avg, rlosses.avg, tlosses.avg ))

        if steps % 1 == 0:
            file_path = os.path.join(args.weight_dir, 'weights-{:07d}.pth.tar'.format(steps))
            state = {
            'epoch' :epoch+1,
            'state_dict' :model.state_dict(),
            'optimizer' : optimizer.state_dict(),
            }
            torch.save(state, file_path)


def reshape(x):
    t = np.array(x, dtype = np.float32)
    return t.reshape(-1, 1)

def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        # this will apply to each layer
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('conv')!=-1 or classname.find('Linear')!=-1):
            if init_type=='normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')#good for relu
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)

            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)
    #print('initialize network with %s' % init_type)
    net.apply(init_func)

class MultiBoxLoss(nn.Module):
    def __init__(self):
        super(MultiBoxLoss, self).__init__()

    def forward(self, predictions, targets):
        print('+++++++++++++++++++++++++++++++++++++++++++++++++++')
        cout, rout = predictions
        """ class """
        class_pred, class_target = cout, targets[:, 0].long()
        pos_index , neg_index    = list(np.where(class_target.cpu().numpy() == 1)[0]), list(np.where(class_target.cpu().numpy() == 0)[0])
        pos_num, neg_num         = len(pos_index), len(neg_index)
        class_pred, class_target = class_pred[pos_index + neg_index], class_target[pos_index + neg_index]

        closs = F.cross_entropy(class_pred, class_target, size_average=False, reduce=False)
        closs = torch.div(torch.sum(closs), 64)

        """ regression """
        reg_pred = rout
        reg_target = targets[:, 1:]
        rloss = F.smooth_l1_loss(reg_pred, reg_target, size_average=False, reduce=False) #1445, 4
        rloss = torch.div(torch.sum(rloss, dim = 1), 4)
        rloss = torch.div(torch.sum(rloss[pos_index]), 16)

        loss = closs + rloss
        return closs, rloss, loss, reg_pred, reg_target, pos_index, neg_index

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def adjust_learning_rate(lr, optimizer, epoch, gamma=0.1):
    """Sets the learning rate to the initial LR decayed 0.9 every 50 epochs"""
    lr = lr * (0.9 ** (epoch // 1))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

if __name__ == '__main__':
    main()


