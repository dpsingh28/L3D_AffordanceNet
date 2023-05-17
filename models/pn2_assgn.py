import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.ops import sample_farthest_points, ball_query
from sklearn.neighbors import NearestNeighbors

class PointNet_Estimation(nn.Module):
    def __init__(self,num_class=3, normal_channel=3):
        super(PointNet_Estimation, self).__init__()
        
        self.normal_channel = normal_channel
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=3, mlp=[64, 64, 128], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256], group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + 3, mlp=[256, 512, 1024], group_all=True)
        
        self.layers_end = nn.Sequential(nn.Linear(1024,1024), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.4),
                                        nn.Linear(1024,512))
        
    def forward(self, xyz):
        B, _, _ = xyz.shape
        # if self.normal_channel:
        #     norm = xyz[:, 3:, :]
        #     xyz = xyz[:, :3, :]
        # else:
        norm = None
        l1_xyz, l1_points = self.sa1(xyz, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        x = l3_points.view(B, 1024)
        x = self.layers_end(x)
        x = F.log_softmax(x, -1)

        return x

class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, kernel_size=1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        self.group_all = group_all

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        # TODO: permute the points to get [B, C, N] shape
        # xyz = xyz.permute(0,2,1)
        # if points is not None:
        #     points = points.permute(0,2,1)
        #     print("points.shape: ",points.shape)

        # print("xyz.shape: ",xyz.shape)
        
        if not self.group_all:
            new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points)
        else:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        
        # print("new_xyz.shape: ",new_xyz.shape)
        # print("new_points.shape: ",new_points.shape)
        
        # TODO: permute again to get new_point as [B, C+D, nsample,npoint]
        new_points = new_points.permute(0, 3, 2, 1) # [B, C+D, nsample,npoint]
        
        # print("new_xyz.shape: ",new_xyz.shape)
        # print("new_points.shape: ",new_points.shape,"\n\n")
        
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points =  F.relu(bn(conv(new_points)))
            # TODO: apply conv and bn from self.mlp_convs and self.mlp_bns

        new_points = torch.max(new_points, 2)[0]
        new_points = new_points.permute(0, 2, 1)
        # new_xyz = new_xyz.permute(0, 2, 1)
        return new_xyz, new_points
    
def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    # print("type(idx): ",type(idx))
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def sample_and_group(npoint, radius, nsample, xyz, points, returnfps=False):
    """
    Input:
        npoint:
        radius:
        nsample:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, npoint, nsample, 3]
        new_points: sampled points data, [B, npoint, nsample, 3+D]
    """
    B, N, C = xyz.shape
    # print(xyz.shape)
    S = npoint
    _,fps_idx = sample_farthest_points(xyz,K=npoint) # [B, npoint, C]
    # print("type(fps_idx): ",type(fps_idx))
    # print(fps_idx.shape)

    new_xyz = index_points(xyz, fps_idx)
    # print(new_xyz.shape)
    _,idx,_ = ball_query(p1=new_xyz, p2=xyz, radius=radius,K=nsample)
    # print(idx.shape)
    grouped_xyz = index_points(xyz, idx) # [B, npoint, nsample, C]
    
    # print(grouped_xyz.shape)
    # print(new_xyz.shape)
    # print(new_xyz.view(B, S, 1, C).shape)
    
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)

    if points is not None:
        grouped_points = index_points(points, idx)
        # print(grouped_xyz_norm.shape)
        # print(grouped_points.shape)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1) # [B, npoint, nsample, C+D]
        # print(new_points.shape)
        # print("here")
    else:
        new_points = grouped_xyz_norm

    return new_xyz, new_points


def sample_and_group_all(xyz, points):
    """
    Input:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, 3]
        new_points: sampled points data, [B, 1, N, 3+D]
    """
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C).to(device)
    grouped_xyz = xyz.view(B, 1, N, C)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points
