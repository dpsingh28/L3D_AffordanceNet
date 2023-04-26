from torch import mul
import torch
import torch.nn as nn
import torch.nn.functional as F
from time import time
import numpy as np
from .pointnet_util import PointNetSetAbstractionMsg, PointNetSetAbstraction, PointNetFeaturePropagation


class PointNet_Estimation(nn.Module):
    def __init__(self, args, num_classes, normal_channel=False):
        super(PointNet_Estimation, self).__init__()
        if normal_channel:
            additional_channel = 3
        else:
            additional_channel = 0
        self.normal_channel = normal_channel
        self.sa1 = PointNetSetAbstractionMsg(512, [0.1, 0.2, 0.4], [
                                             32, 64, 128], 3+additional_channel, [[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        self.sa2 = PointNetSetAbstractionMsg(
            128, [0.4, 0.8], [64, 128], 128+128+64, [[128, 128, 256], [128, 196, 256]])
        self.sa3 = PointNetSetAbstraction(
            npoint=None, radius=None, nsample=None, in_channel=512 + 3, mlp=[256, 512, 1024], group_all=True)

        self.fp3 = PointNetFeaturePropagation(in_channel=1536, mlp=[256, 256])
        self.fp2 = PointNetFeaturePropagation(in_channel=576, mlp=[256, 128])
        self.fp1 = PointNetFeaturePropagation(
            in_channel=134+additional_channel, mlp=[128, 512])

        # self.classifier = nn.ModuleList()
        # for i in range(num_classes):
        #     classifier = nn.Sequential(
        #         nn.Conv1d(1024, 512, 1),
        #         nn.BatchNorm1d(512),
        #         nn.Dropout(0.5),
        #         nn.Conv1d(512, 1, 1)
        #     )
        #     self.classifier.append(classifier)

        # self.postconv = nn.Conv1d(512,512,1)
        # self.bn = nn.BatchNorm1d(512)

    def forward(self, xyz,text_emb):
        # Set Abstraction layers
        xyz = xyz.contiguous()
        # print("pcd shape",xyz.size())
        B, C, N = xyz.shape
        if self.normal_channel:
            l0_points = xyz
            l0_xyz = xyz[:, :3, :]
        else:
            # l0_points = xyz.transpose(1, 2).contiguous()
            # l0_points = None
            l0_xyz = xyz
            l0_points = xyz

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        # print(l1_xyz.size(), l1_points.size())
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        # print(l2_xyz.size(), l2_points.size())
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        # print(l3_xyz.size())
        # print(l3_points.size())
        # Feature Propagation layers
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        # print(l2_points.size())
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        # print("l1",l1_points.size())
        l0_points = self.fp1(l0_xyz, l1_xyz, torch.cat(
            [l0_xyz, l0_points], 1), l1_points)
        
        # l0_points  = self.bn(l0_points)

        # l0_points = self.bn(self.postconv(l0_points))

        # clip_align_tensor = l0_points.unsqueeze(-1).repeat(1,1,1,18).permute(0,2,3,1)

        # print(l0_points.shape)
        # print(clip_align_tensor.shape)

        return l0_points
        
        # text_emb = text_emb.repeat(xyz.shape[0],xyz.shape[2],1).permute(0,2,1)
        # # print(text_emb.shape,l0_points.shape)
        # aggregated_vec = torch.concat([l0_points,text_emb],dim=1)

        # # print("aggregated",aggregated_vec.shape)

        # # FC layers
        # score = self.classifier[0](aggregated_vec)
        # # print(score.shape)
        # score = torch.sigmoid(score)
        # return score

if __name__ == '__main__':
    import torch
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = '4'
    model = PointNet_Estimation(18).cuda()
    xyz = torch.rand(6, 3, 4096).cuda()
    print(model(xyz).size())
