# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
from torch import nn
import torch
import math
import torch.nn.functional as F


from RelativeTransformer.RTlayers import SRT
from RelativeTransformer.RTlayers import TRT

def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod

class STRT(nn.Module):
    def __init__(self, num_classes, dataset, seg, args, n_layers, n_head, d_k, d_v, d_in,
            d_model, num=4, dropout=0.1, stream='joint', bias=True):
        super(STRT, self).__init__()

        self.dataset = dataset
        self.seg = seg
        self.stream = stream

        if dataset == 'UAV':
            # Human-UAV
            num_joint = 17
            dim = 2
        else:
            # NTU60 or NTU120
            num_joint = 25
            dim = 3

        self.d = dim


        bs = int(args.batch_size/num)
        if args.train:
            self.spa = self.one_hot(bs, num_joint, self.seg)    #[64,20,25,25]
            self.spa = self.spa.permute(0, 3, 2, 1).cuda()   #[64,25,25,20]
            self.tem = self.one_hot(bs, self.seg, num_joint)   #[64,25,20,20]
            self.tem = self.tem.permute(0, 3, 1, 2).cuda()   #[64,20,20,25]
        else:
            self.spa = self.one_hot(8*5, num_joint, self.seg)
            self.spa = self.spa.permute(0, 3, 2, 1).cuda()
            self.tem = self.one_hot(8*5, self.seg, num_joint)
            self.tem = self.tem.permute(0, 3, 1, 2).cuda()


        self.joint_embed = embed(num_joint, dim, 256, norm=True, bias=bias)
        self.adaptive_embed = embed(num_joint, dim, 256, norm=True, bias=bias)
        self.short_embed = embed(num_joint, dim, 128, norm=True, bias=bias)
        self.long_embed = embed(num_joint, dim, 128, norm=True, bias=bias)
        self.tem_embed = embed(num_joint, self.seg, 768, norm=False, bias=bias)
        self.spa_embed = embed(num_joint, num_joint, 768, norm=False, bias=bias)

        self.maxpool = nn.AdaptiveMaxPool2d((1, 1))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(768, num_classes)


        self.input_emb = nn.Conv2d(128, d_model, kernel_size=1)
        self.spa_enc = SRT(d=num_joint,
                            hidden_size=d_model,
                            num_layers=n_layers,
                            num_head=n_head,
                            dk=d_k,
                            dv=d_v,
                            d_in=d_in,
                            dropout=dropout)

        self.tem_enc = TRT(hidden_size=d_model,
                            num_layers=n_layers,
                            num_head=n_head,
                            dk=d_k,
                            dv=d_v,
                            d_in=d_in,
                            dropout=dropout)

        # inherent connection
        if dim == 2:
            # UAV
            self.paris = (
                (1, 2), (1, 3), (2, 4), (3, 5), (1, 6), (1, 7), (6, 8), (6, 12), (7, 9), (7, 13), (8, 10),
                (9, 11), (12, 14), (13, 15), (14, 16), (15, 17)
            )
        else:
            # NTU
            self.paris = (
                (1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6), (8, 7), (9, 21), (10, 9), (11, 10), (12, 11),
                (13, 1),
                (14, 13), (15, 14), (16, 15), (17, 1), (18, 17), (19, 18), (20, 19), (22, 23), (21, 21), (23, 8), (24, 25),
                (25, 12)
            )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))


    def forward(self, input):

        bs, step, dim = input.size()
        device = input.device

        if self.d == 2:
            # UAV
            num_joints = dim // 6
            input = input.view((bs, step, num_joints, 6))
        else:
            # NTU
            num_joints = dim // 9
            input = input.view((bs, step, num_joints, 9))

        input = input.permute(0, 3, 2, 1).contiguous()

        # choose joint or bone stream
        if self.stream == 'bone':
            bone = input  #[64, 3,25,20]
            for v1, v2 in self.paris:
                v1 -= 1
                v2 -= 1
                bone[:, :, v1, :] = input[:, :, v1, :] - input[:, :, v2, :]

            input = F.normalize(bone, p=2, dim=1)



        x1, x2, x3 = input[:, 0:2, :, :], input[:, 2:4, :, :], input[:, 4:6, :, :]

        # fixed motion
        short = x2-x1
        long = x3-x1

        # adaptive motion
        adap = x1[:, :, :, 1:] - x1[:, :, :, 0:-1]    #[64, 3, 25, 19]
        adaptive = torch.cat([adap.new(bs, adap.size(1), num_joints, 1).zero_(), adap], dim=-1)    #[64, 3, 25, 20]

        pos = self.joint_embed(x1)
        tem = self.tem_embed(self.tem.cuda(device))
        spa = self.spa_embed(self.spa.cuda(device))

        adaptive = self.adaptive_embed(adaptive)
        short = self.short_embed(short)
        long = self.long_embed(long)

        dy = torch.cat([pos, adaptive, short, long], 1)

        # spatial encoding
        output_spa = self.spa_enc(dy + spa)


        # temporal encoding
        output_tem = self.tem_enc(output_spa + tem)
        output = output_tem.permute(0, 2, 1, 3).contiguous()

        # Classification
        B, C, V, T = output.size()
        output = output.contiguous().view(B, C, -1)
        output = output.mean(2)
        output = self.fc(output)

        return output

    def one_hot(self, bs, spa, tem):

        y = torch.arange(spa).unsqueeze(-1)
        y_onehot = torch.FloatTensor(spa, spa)

        y_onehot.zero_()
        y_onehot.scatter_(1, y, 1)

        y_onehot = y_onehot.unsqueeze(0).unsqueeze(0)
        y_onehot = y_onehot.repeat(bs, tem, 1, 1)

        return y_onehot


class norm_data(nn.Module):
    def __init__(self, num_joint, dim= 64):
        super(norm_data, self).__init__()


        self.bn = nn.BatchNorm1d(dim * num_joint)

    def forward(self, x):
        bs, c, num_joints, step = x.size()
        x = x.view(bs, -1, step)
        x = self.bn(x)
        x = x.view(bs, -1, num_joints, step).contiguous()
        return x

class embed(nn.Module):
    def __init__(self, num_joint, dim = 3, dim1 = 128, norm = True, bias = False):
        super(embed, self).__init__()

        if norm:
            self.cnn = nn.Sequential(
                norm_data(num_joint, dim),
                nn.Conv2d(dim, 64, kernel_size=1, bias=bias),
                nn.ReLU(),
                nn.Conv2d(64, dim1, kernel_size=1, bias=bias),
                nn.ReLU(),
            )
        else:
            self.cnn = nn.Sequential(
                nn.Conv2d(dim, 64, kernel_size=1, bias=bias),
                nn.ReLU(),
                nn.Conv2d(64, dim1, kernel_size=1, bias=bias),
                nn.ReLU(),
            )

    def forward(self, x):
        x = self.cnn(x)
        return x