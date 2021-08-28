# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import argparse
import time
import shutil
import os

import os.path as osp
import csv
import numpy as np

import torch
import random

seed=1337
np.random.seed(seed)
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.cuda.empty_cache()

import torch
torch.set_printoptions(precision=None, threshold=10000, edgeitems=None, linewidth=None, profile=None)
import torch.nn as nn
import torch.optim as optim

from RelativeTransformer.model import STRT
from data import NTUDataLoaders, AverageMeter
import fit
from util import make_dir, get_num_classes
from Optim import ScheduledOptim

parser = argparse.ArgumentParser(description='Skeleton-Based Action Recgnition')
fit.add_fit_args(parser)
parser.set_defaults(
    network='UAV_j20',
    dataset='UAV',
    case=0,
    batch_size=128,
    start_epoch=0,
    max_epochs=50,
    monitor='val_acc',
    lr=0.0005,
    weight_decay=0.0001,
    lr_factor=0.1,
    workers=16,
    print_freq=20,
    train=0,
    seg=20,
    stream='joint',
    n_layers=3,
    n_head=8,
    d_k=96,
    d_v=96,
    d_in=1024,
    d_model=768,
    n_warmup_steps=700,
    warmup_learning_rate=4e-07,
    hold_base_rate_steps=0,
    decay_rate=0.9996,
    dropout=0.1,
    work_dir='./work_dir/UAV_j20',
    gpus='0,1,2,3',
    weights=0,
    )
args = parser.parse_args()

def main():
    torch.backends.cudnn.enabled = False
    args.num_classes = get_num_classes(args.dataset)
    gpus = [int(i) for i in args.gpus.split(',')]
    assert len(gpus) > 0, "args.gpus length must be > 0"
    print("gpus:", gpus, type(gpus))
    device = torch.device("cuda:{}".format(gpus[0]) if torch.cuda.is_available() else 'cpu')
    print_log("device:{}".format(device))
    model = STRT(args.num_classes, args.dataset, args.seg, args, args.n_layers, args.n_head,
                 args.d_k, args.d_v, args.d_in, args.d_model, len(gpus), args.dropout, args.stream)
    model = nn.DataParallel(model, device_ids=gpus).to(device)
    gpu = gpus[0]

    # NTU Data loading
    ntu_loaders = NTUDataLoaders(args.dataset, args.case, seg=args.seg)
    train_loader = ntu_loaders.get_train_loader(args.batch_size, args.workers)
    val_loader = ntu_loaders.get_val_loader(args.batch_size, args.workers)
    train_size = ntu_loaders.get_train_size()
    val_size = ntu_loaders.get_val_size()


    # create checkpoint
    best_epoch = 0
    output_dir = make_dir(args.dataset)

    save_path = os.path.join(output_dir, args.network)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    checkpoint = osp.join(save_path, '%s_best.pth' % args.case)
    earlystop_cnt = 0
    csv_file = osp.join(save_path, '%s_log.csv' % args.case)
    log_res = list()

    lable_path = osp.join(save_path, '%s_lable.txt' % args.case)
    pred_path = osp.join(save_path, '%s_pred.txt' % args.case)



    total = get_n_params(model)
    # print(model)
    # print_log(model)
    # print('The number of parameters: ', total)
    print_log('The number of parameters:{} '.format(total))
    # print('The modes is:{}'.format(args.network))
    print_log('The modes is:{}'.format(args.network))
    print_log('max_lr:{} n_layers:{} n_warmup_steps:{} warmup_learning_rate:{} hold_base_rate_steps:{} decay_rate:{} seg:{}'.
              format(args.lr, args.n_layers, args.n_warmup_steps, args.warmup_learning_rate, args.hold_base_rate_steps, args.decay_rate, args.seg)
    )

    if torch.cuda.is_available():
        print('It is using GPU!')
        model = model.cuda(gpu)

    global_step = 0
    current_epoch = 0
    if args.weights == 1:
        current_epoch = torch.load(checkpoint)['epoch'] - 1
        global_step = current_epoch * train_size//args.batch_size
        print("global_step:", global_step)


    criterion = LabelSmoothingLoss(args.num_classes, smoothing=0.1).cuda(gpu)
    # optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-09)
    Optim = ScheduledOptim(optimizer, args.lr, args.d_model, args.n_warmup_steps, args.max_epochs,
                           args.warmup_learning_rate, args.hold_base_rate_steps, args.batch_size, args.decay_rate, args.work_dir, global_step)

    if args.monitor == 'val_acc':
        mode = 'max'
        monitor_op = np.greater
        best = -np.Inf
        str_op = 'improve'
    elif args.monitor == 'val_loss':
        mode = 'min'
        monitor_op = np.less
        best = np.Inf
        str_op = 'reduce'

    test_loader = ntu_loaders.get_test_loader(8*4, args.workers)

    # print('Train on %d samples, validate on %d samples' % (train_size, val_size))
    print_log('Train on %d samples, validate on %d samples' % (train_size, val_size))



    # Training
    if args.train ==1:
        for epoch in range(current_epoch, args.max_epochs):

            print_log('lr: {}'.format(optimizer.param_groups[0]['lr']))

            t_start = time.time()
            train_loss, train_acc = train(train_loader, model, criterion, Optim, epoch, gpu, args.weights, checkpoint)
            val_loss, val_acc = validate(val_loader, model, criterion)
            log_res += [[train_loss, train_acc.cpu().numpy(),\
                         val_loss, val_acc.cpu().numpy()]]

            print_log('Epoch-{:<3d} {:.1f}s\t'
                  'Train: loss {:.4f}\taccu {:.4f}\tValid: loss {:.4f}\taccu {:.4f}'
                  .format(epoch + 1, time.time() - t_start, train_loss, train_acc, val_loss, val_acc))


            current = val_loss if mode == 'min' else val_acc

            ####### store tensor in cpu
            current = current.cpu()

            if monitor_op(current, best):
                print_log('Epoch %d: %s %sd from %.4f to %.4f, '
                      'saving model to %s'
                      % (epoch + 1, args.monitor, str_op, best, current, checkpoint))
                print_log('Current time: {:.2f}s'.format(time.time()))
                best = current
                best_epoch = epoch + 1
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'best': best,
                    'monitor': args.monitor,
                    'optimizer': optimizer.state_dict(),
                }, checkpoint)
                earlystop_cnt = 0
            else:
                print_log('Epoch %d: %s did not %s' % (epoch + 1, args.monitor, str_op))
                print_log('Current time: {:.2f}s'.format(time.time()))
                earlystop_cnt += 1

            # scheduler.step()

        print_log('Best %s: %.4f from epoch-%d' % (args.monitor, best, best_epoch))
        with open(csv_file, 'w') as fw:
            cw = csv.writer(fw)
            cw.writerow(['loss', 'acc', 'val_loss', 'val_acc'])
            cw.writerows(log_res)
        print_log('Save train and validation log into into %s' % csv_file)

    ### Test
    args.train = 0
    gpus = [int(i) for i in args.gpus.split(',')]
    model = STRT(args.num_classes, args.dataset, args.seg, args,
                args.n_layers, args.n_head, args.d_k, args.d_v, args.d_in, args.d_model, len(gpus))

    model = nn.DataParallel(model, device_ids=gpus).to(device)
    test(test_loader, model, checkpoint, lable_path, pred_path)


def train(train_loader, model, criterion, optimizer, epoch, gpu, weights, checkpoint):
    losses = AverageMeter()
    acces = AverageMeter()
    if weights == 1:
        model.load_state_dict(torch.load(checkpoint)['state_dict'])

    model.train()

    for i, (inputs, target) in enumerate(train_loader):

        # optimizer.zero_grad()
        output = model(inputs.cuda(gpu))
        target = target.cuda(async = True)
        loss = criterion(output, target).cuda(gpu)
        inputs = inputs.cuda(async = True)

        # measure accuracy and record loss
        acc, _, _ = accuracy(output.data, target)
        losses.update(loss.item(), inputs.size(0))
        acces.update(acc[0], inputs.size(0))

        # backward
        optimizer.zero_grad()  # clear gradients out before each mini-batch
        loss.backward()
        # optimizer.step()
        optimizer.step_and_update_lr()

        if (i + 1) % args.print_freq == 0:
            print_log('Epoch-{:<3d} {:3d} batches\t'
                  'loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'accu {acc.val:.3f} ({acc.avg:.3f})'.format(epoch + 1, i + 1, loss=losses, acc=acces))

    return losses.avg, acces.avg


def validate(val_loader, model, criterion):
    losses = AverageMeter()
    acces = AverageMeter()
    model.eval()

    for i, (inputs, target) in enumerate(val_loader):
        with torch.no_grad():
            output = model(inputs.cuda())
        target = target.cuda(async=True)
        with torch.no_grad():
            loss = criterion(output, target)

        # measure accuracy and record loss
        acc,_,_ = accuracy(output.data, target)
        losses.update(loss.item(), inputs.size(0))
        acces.update(acc[0], inputs.size(0))

    return losses.avg, acces.avg





def test(test_loader, model, checkpoint, lable_path, pred_path):

    acces = AverageMeter()
    # load learnt model that obtained best performance on validation set
    # model.load_state_dict(torch.load(checkpoint)['state_dict'])
    dict = torch.load(checkpoint)['state_dict']
    # for key in list(dict.keys()):
    #     if key == 'module.spa_enc.ring_att_spa.0.WO.weight':
    #         del dict[key]
    #     if key == 'module.spa_enc.ring_att_spa.1.WO.weight':
    #         del dict[key]
    #     if key == 'module.spa_enc.ring_att_spa.2.WO.weight':
    #         del dict[key]
    #     if key == 'module.spa_enc.star_att_spa.0.WO.weight':
    #         del dict[key]
    #     if key == 'module.spa_enc.star_att_spa.1.WO.weight':
    #         del dict[key]
    #     if key == 'module.spa_enc.star_att_spa.2.WO.weight':
    #         del dict[key]
    #     if key == 'module.tem_enc.ring_att_spa.0.WO.weight':
    #         del dict[key]
    #     if key == 'module.tem_enc.ring_att_spa.1.WO.weight':
    #         del dict[key]
    #     if key == 'module.tem_enc.ring_att_spa.2.WO.weight':
    #         del dict[key]
    #     if key == 'module.tem_enc.star_att_spa.0.WO.weight':
    #         del dict[key]
    #     if key == 'module.tem_enc.star_att_spa.1.WO.weight':
    #         del dict[key]
    #     if key == 'module.tem_enc.star_att_spa.2.WO.weight':
    #         del dict[key]
    # torch.save(dict, './model_deleted.pth')
    model.load_state_dict(dict)
    model.eval()

    label_output = list()
    pred_output = list()

    # Human-UAV
    error = torch.zeros(1, 155)
    count = torch.zeros(1, 155)
    err_pre = torch.zeros(155, 155)

    ## NTU60
    # error = torch.zeros(1, 60)
    # count = torch.zeros(1, 60)
    # err_pre = torch.zeros(60, 60)

    ## NTU120
    # error = torch.zeros(1, 120)
    # count = torch.zeros(1, 120)
    # err_pre = torch.zeros(120, 120)

    t_start = time.time()
    for i, (inputs, target) in enumerate(test_loader):
        with torch.no_grad():
            output = model(inputs.cuda())
            output = output.view((-1, inputs.size(0)//target.size(0), output.size(1)))
            output = output.mean(1)

        label_output.append(target.cpu().numpy())
        pred_output.append(output.cpu().numpy())

        acc, correct, pred = accuracy(output.data, target.cuda(async=True))

        for i in range(len(correct[0])):
            count[0][target[i]] += 1
            if correct[0][i] == False:
                error[0][target[i]] += 1
                err_pre[target[i]][pred[0][i]] += 1

        acces.update(acc[0], inputs.size(0))

    print("count:", count)
    print("error:", error)
    print("err_pre:", err_pre)

    print_log('count:{}'.format(count))
    print_log('error:{}'.format(error))
    print_log('err_pre:{}'.format(err_pre))


    torch.save(err_pre, 'save.pt')

    label_output = np.concatenate(label_output, axis=0)
    np.savetxt(lable_path, label_output, fmt='%d')
    pred_output = np.concatenate(pred_output, axis=0)
    np.savetxt(pred_path, pred_output, fmt='%f')

    # print('Test: accuracy {:.3f}, time: {:.2f}s'
    #       .format(acces.avg, time.time() - t_start))
    print_log('Test: accuracy {:.3f}, time: {:.2f}s'
          .format(acces.avg, time.time() - t_start))


def print_log(str):
    print(str)
    if not os.path.exists(args.work_dir):
        os.makedirs(args.work_dir)
    with open('{}/log.txt'.format(args.work_dir), 'a') as f:
        print(str, file=f)

def accuracy(output, target):
    batch_size = target.size(0)
    _, pred = output.topk(1, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    correct_c = correct.view(-1).float().sum(0, keepdim=True)

    return correct_c.mul_(100.0 / batch_size), correct, pred

def save_checkpoint(state, filename='checkpoint.pth.tar', is_best=False):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.0, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim


    def forward(self, pred, target):
        # pred = pred.log_softmax(dim=self.dim)
        pred = pred.softmax(dim=self.dim)
        pred = torch.log(pred+1e-10)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))

if __name__ == '__main__':
    main()
    
