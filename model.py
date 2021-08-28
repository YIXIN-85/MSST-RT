# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
from torch import nn
import torch
import math
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from graph.ntu_rgb_d import Graph
from torch.autograd import Variable

def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod

class SGN(nn.Module):
    def __init__(self, num_classes, dataset, seg, args, bias = True):
        super(SGN, self).__init__()

        # graph
        self.graph = Graph(labeling_mode='spatial')
        A = self.graph.A
        self.PA = nn.Parameter(torch.from_numpy(A.astype(np.float32)))
        nn.init.constant_(self.PA, 1e-6)
        self.A = Variable(torch.from_numpy(A.astype(np.float32)), requires_grad=False)

        self.dim1 = 256
        self.dataset = dataset
        self.seg = seg
        num_joint = 25
        bs = args.batch_size
        if args.train:
            self.spa = self.one_hot(bs, num_joint, self.seg)    #[64,20,25,25]
            self.spa = self.spa.permute(0, 3, 2, 1).cuda()   #[64,25,25,20]
            self.tem = self.one_hot(bs, self.seg, num_joint)   #[64,25,20,20]
            self.tem = self.tem.permute(0, 3, 1, 2).cuda()   #[64,20,20,25]
        else:
            self.spa = self.one_hot(16 * 5, num_joint, self.seg)
            self.spa = self.spa.permute(0, 3, 2, 1).cuda()
            self.tem = self.one_hot(16 * 5, self.seg, num_joint)
            self.tem = self.tem.permute(0, 3, 1, 2).cuda()

        self.tem_embed = embed(self.seg, 64*4, norm=False, bias=bias)
        self.spa_embed = embed(num_joint, 64, norm=False, bias=bias)
        self.joint_embed = embed(3, 64, norm=True, bias=bias)
        self.dif_embed = embed(3, 64, norm=True, bias=bias)
        # self.bone_embed = embed(3, 64, norm=True, bias=bias)
        self.maxpool = nn.AdaptiveMaxPool2d((1, 1))
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.cnn = local(self.dim1, self.dim1 * 2, bias=bias)
        self.tam = TAM(self.dim1, self.dim1 *2, bias=bias)
        self.compute_g1 = compute_g_spa(self.dim1 // 2, self.dim1, bias=bias)
        # self.compute_g2 = compute_g_spa(self.dim1 // 2, self.dim1, bias=bias)
        # self.compute_g3 = compute_g_spa(self.dim1 , self.dim1, bias=bias)
        self.gcn1 = gcn_spa(self.dim1 // 2, self.dim1 // 2, bias=bias)
        self.gcn2 = gcn_spa(self.dim1 // 2, self.dim1, bias=bias)
        self.gcn3 = gcn_spa(self.dim1, self.dim1, bias=bias)
        self.fc = nn.Linear(self.dim1 * 2 , num_classes)
        # self.paris = (
        #     (1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6), (8, 7), (9, 21), (10, 9), (11, 10), (12, 11),
        #     (13, 1),
        #     (14, 13), (15, 14), (16, 15), (17, 1), (18, 17), (19, 18), (20, 19), (22, 23), (21, 21), (23, 8), (24, 25),
        #     (25, 12)
        # )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

        nn.init.constant_(self.gcn1.w.cnn.weight, 0)
        nn.init.constant_(self.gcn2.w.cnn.weight, 0)
        nn.init.constant_(self.gcn3.w.cnn.weight, 0)


    def forward(self, input):
        # A = self.A.cuda(input.get_device())
        # A = A + self.PA
        # Dynamic Representation
        bs, step, dim = input.size()    #[64, 20, 75]
        num_joints = dim // 3
        input = input.view((bs, step, num_joints, 3))

        # (bs,3,num_joints,step)
        input = input.permute(0, 3, 2, 1).contiguous()  #[64, 3,25,20]
        dif = input[:, :, :, 1:] - input[:, :, :, 0:-1]    #[64, 3, 25, 19]
        dif = torch.cat([dif.new(bs, dif.size(1), num_joints, 1).zero_(), dif], dim=-1)    #[64, 3, 25, 20]
        # bone = input
        # for v1, v2 in self.paris:
        #     v1 -= 1
        #     v2 -= 1
        #     bone[:, :, v1, :] = input[:, :, v1, :]-input[:, :, v2, :]
        # bone = self.bone_embed(bone)
        pos = self.joint_embed(input)    #[64, 64, 25, 20]
        tem1 = self.tem_embed(self.tem)    # self.tem:[64, 20, 20, 25]   tem1:[64, 256, 25, 20]
        spa1 = self.spa_embed(self.spa)    # self.spa:[64, 25, 25, 20]   spa1:[64, 64, 25, 20]
        dif = self.dif_embed(dif)    #[64, 64, 25, 20]
        dy = pos + dif
        # Joint-level Module
        input = torch.cat([dy, spa1], 1)    #[64, 128, 25, 20]
        # g = self.compute_g1(input)
        # print("g:",g.type())
        # print("graph:",graph.type())
        g1 = self.compute_g1(input)
        g = g1
        input = self.gcn1(input, g)
        # g = self.compute_g2(input)
        input = self.gcn2(input, g)
        # g = self.compute_g3(input)
        input = self.gcn3(input, g)
        # Frame-level Module
        input = input + tem1
        input = self.cnn(input)
        # input = self.tam(input)
        # Classification
        output = self.maxpool(input)
        output = torch.flatten(output, 1)
        output = self.fc(output)

        return output

    def one_hot(self, bs, spa, tem):
        # spa: [64, 25, 20];
        # tem: [64, 20, 25];

        y = torch.arange(spa).unsqueeze(-1)   #[25,1]
        y_onehot = torch.FloatTensor(spa, spa)

        y_onehot.zero_()
        y_onehot.scatter_(1, y, 1)

        y_onehot = y_onehot.unsqueeze(0).unsqueeze(0)
        y_onehot = y_onehot.repeat(bs, tem, 1, 1)

        return y_onehot






class norm_data(nn.Module):
    def __init__(self, dim= 64):
        super(norm_data, self).__init__()

        self.bn = nn.BatchNorm1d(dim* 25)

    def forward(self, x):
        bs, c, num_joints, step = x.size()
        x = x.view(bs, -1, step)
        x = self.bn(x)
        x = x.view(bs, -1, num_joints, step).contiguous()
        return x

class embed(nn.Module):
    def __init__(self, dim = 3, dim1 = 128, norm = True, bias = False):
        super(embed, self).__init__()

        if norm:
            self.cnn = nn.Sequential(
                norm_data(dim),
                cnn1x1(dim, 64, bias=bias),
                nn.ReLU(),
                cnn1x1(64, dim1, bias=bias),
                nn.ReLU(),
            )
        else:
            self.cnn = nn.Sequential(
                cnn1x1(dim, 64, bias=bias),
                nn.ReLU(),
                cnn1x1(64, dim1, bias=bias),
                nn.ReLU(),
            )

    def forward(self, x):
        x = self.cnn(x)
        return x

class cnn1x1(nn.Module):
    def __init__(self, dim1 = 3, dim2 =3, bias = True):
        super(cnn1x1, self).__init__()
        self.cnn = nn.Conv2d(dim1, dim2, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.cnn(x)
        return x

class local(nn.Module):
    def __init__(self, dim1 = 3, dim2 = 3, bias = False):
        super(local, self).__init__()
        self.maxpool = nn.AdaptiveMaxPool2d((1, 20))
        self.avgpool = nn.AdaptiveAvgPool2d((1,20))
        self.cnn1 = nn.Conv2d(dim1, dim1, kernel_size=(1, 3), padding=(0, 1), bias=bias)
        self.bn1 = nn.BatchNorm2d(dim1)
        self.relu = nn.ReLU()
        self.cnn2 = nn.Conv2d(dim1, dim2, kernel_size=1, bias=bias)
        self.bn2 = nn.BatchNorm2d(dim2)
        self.dropout = nn.Dropout2d(0.2)

    def forward(self, x1):
        e_x = torch.sum(torch.exp(x1),dim=1,keepdim=True)
        x1 = self.avgpool(x1.mul(e_x)).mul_(26).div_(self.avgpool(e_x)).mul_(26)
        # x1 = self.maxpool(x1)
        x = self.cnn1(x1)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.cnn2(x)
        x = self.bn2(x)
        x = self.relu(x)

        return x

class TAM(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 bias=False,
                 n_segment=20,
                 kernel_size=3,
                 stride=1,
                 padding=1):
        super(TAM, self).__init__()
        self.in_channels = in_channels
        self.n_segment = n_segment
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        print('TAM with kernel_size {}.'.format(kernel_size))

        self.G = nn.Sequential(
            nn.Linear(n_segment, n_segment * 2, bias=False),
            nn.BatchNorm1d(n_segment * 2), nn.ReLU(inplace=True),
            nn.Linear(n_segment * 2, kernel_size, bias=False), nn.Softmax(-1))

        self.L = nn.Sequential(
            nn.Conv1d(in_channels,
                      in_channels // 4,
                      kernel_size,
                      stride=1,
                      padding=kernel_size // 2,
                      bias=False), nn.BatchNorm1d(in_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels // 4, in_channels, 1, bias=False),
            nn.Sigmoid())

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU()
        self.cnn2 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(0.2)

    def forward(self, x):
        # [64, 128, 25, 20]
        N, C, V, T = x.size()
        new_x = x.permute(0, 1, 3, 2).contiguous()
        out = F.adaptive_avg_pool1d(new_x.view(N*C,T,V),1)
        out = out.view(-1,T)
        conv_kernel = self.G(out.view(-1,T)).view(N * C, 1, -1, 1)
        local_activation = self.L(out.view(N, C, T)).view(N, C, T, 1)
        new_x = new_x + local_activation
        out = F.conv2d(new_x.view(1, N * C, T, V),
                       conv_kernel,
                       bias=None,
                       stride=(self.stride, 1),
                       padding=(self.padding, 0),
                       groups=N * C)
        out = out.view(N, C, T, V)
        out = out.permute(0, 1, 3, 2).contiguous()
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.cnn2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out







class gcn_spa(nn.Module):
    def __init__(self, in_feature, out_feature, bias = False):
        super(gcn_spa, self).__init__()
        self.bn = nn.BatchNorm2d(out_feature)
        self.relu = nn.ReLU()
        self.conv_value = cnn1x1(in_feature, in_feature)
        self.w = cnn1x1(in_feature, out_feature, bias=False)
        self.w1 = cnn1x1(in_feature, out_feature, bias=bias)


    def forward(self, x1, g): #[64, 128, 25, 20]
        # x1 = self.conv_value(x1)
        x = x1.permute(0, 3, 2, 1).contiguous()
        x = g.matmul(x)
        x = x.permute(0, 3, 2, 1).contiguous()
        x = self.w(x) + self.w1(x1)
        x = self.relu(self.bn(x))
        return x

class compute_g_spa(nn.Module):
    def __init__(self, dim1 = 64 *3, dim2 = 64*3, bias = False):
        super(compute_g_spa, self).__init__()
        self.dim1 = dim1
        self.dim2 = dim2
        self.g1 = cnn1x1(self.dim1, self.dim2, bias=bias)
        self.g2 = cnn1x1(self.dim1, self.dim2, bias=bias)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x1):

        g1 = self.g1(x1).permute(0, 3, 2, 1).contiguous()
        g2 = self.g2(x1).permute(0, 3, 1, 2).contiguous()
        g3 = g1.matmul(g2)
        g = self.softmax(g3)
        return g

class compute_dnl(nn.Module):
    def __init__(self, dim1 = 64 *3, dim2 = 64*3, bias = False):
        super(compute_g_spa, self).__init__()

        self.dim1 = dim1
        self.dim2 = dim2
        self.scale = math.sqrt(self.dim2)
        self.g1 = cnn1x1(self.dim1, self.dim2, bias=bias) #[64, 256, 25, 20]
        self.g2 = cnn1x1(self.dim1, self.dim2, bias=bias) #[64, 256, 25, 20]
        self.conv_mask = cnn1x1(self.dim1, 1, bias=bias)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x1): #[64, 128, 25, 20]
        N, C, V, T = x1.size()

        g1 = self.g1(x1).permute(0,3,1,2) #[64, 20, 256, 25]
        g2 = self.g2(x1).permute(0,3,1,2) #[64, 20, 256, 25]

        # [N*T, C', V]
        g1 = g1.contiguous().view(-1, g1.size(2), g1.size(3))
        g2 = g2.contiguous().view(-1, g2.size(2), g2.size(3))

        g1_mean = g1.mean(2).unsqueeze(2)
        g2_mean = g2.mean(2).unsqueeze(2)
        g1 -= g1_mean
        g2 -= g2_mean

        # [N*T, V, V]
        sim_map = torch.matmul(g1.transpose(1, 2), g2)
        sim_map = sim_map / self.scale
        # sim_map = self.softmax(sim_map)

        mask = self.conv_mask(x1).permute(0,3,1,2) #[64,20.1,25]
        mask = mask.contiguous().view(-1, mask.size(2), mask.size(3))
        # mask = self.softmax(mask)
        out_sim = self.softmax(sim_map + mask)

        out_sim = (out_sim).contiguous().view(N,T,V,V)

        # g3 = g1.matmul(g2)
        # g = self.softmax(g3)
        return out_sim


