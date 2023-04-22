# Copyright (c) Gorilla-Lab. All rights reserved.
import os
from os.path import join as opj
import numpy as np
import torch
from torch.utils.data import Dataset
import h5py
import json
from utils.provider import rotate_point_cloud_SO3, rotate_point_cloud_y
import pickle as pkl
from ordered_set import OrderedSet
from sklearn.model_selection import train_test_split


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc, centroid, m


def semi_points_transform(points):
    spatialExtent = np.max(points, axis=0) - np.min(points, axis=0)
    eps = 2e-3*spatialExtent[np.newaxis, :]
    jitter = eps*np.random.randn(points.shape[0], points.shape[1])
    points_ = points + jitter
    return points_


class AffordNetDataset(Dataset):
    def __init__(self, data_dir, split, partial=False, rotate='None', semi=False):
        super().__init__()
        self.data_dir = data_dir
        self.split = split

        self.partial = partial
        self.rotate = rotate
        self.semi = semi
        self.num_classes : int = 0
        self.classes = []
        self.classes_set = None

        self.load_data()

        self.affordance = self.all_data[0]["affordance"]

        return

    def load_data(self):
        self.all_data = []
        if self.semi:
            with open(opj(self.data_dir, 'semi_label_1.pkl'), 'rb') as f:
                temp_data = pkl.load(f)
        else:
            if self.partial:
                if self.split == 'train' or self.split == 'test':
                    with open(opj(self.data_dir, 'partial_train_data.pkl'), 'rb') as f:
                        temp_data = pkl.load(f)
                        train_set, test_set = train_test_split(temp_data, test_size=0.14, random_state=42)
                        if self.split == 'train':
                            print('train')
                            temp_data = train_set
                        else:
                            print('test')
                            temp_data = test_set
                else:
                    with open(opj(self.data_dir, 'partial_val_data.pkl'), 'rb') as f:
                        temp_data = pkl.load(f)
            elif self.rotate != "None" and self.split != 'train':
                with open(opj(self.data_dir, 'rotate_%s_data.pkl' % self.split), 'rb') as f:
                    temp_data_rotate = pkl.load(f)
                with open(opj(self.data_dir, 'full_shape_%s_data.pkl' % self.split), 'rb') as f:
                    temp_data = pkl.load(f)
            elif self.split == 'train' or self.split == 'test':
                # 感觉这么写多少有点降智（捂脸），让我思考一下怎么优雅一点
                with open(opj(self.data_dir, 'full_shape_train_data.pkl'), 'rb') as f:
                    temp_data = pkl.load(f)
                train_set, test_set = train_test_split(temp_data, test_size=0.14, random_state=42)
                if self.split == 'train':
                    print('train')
                    temp_data = train_set
                else:
                    print('test')
                    temp_data = test_set
            else:
                print('val')
                with open(opj(self.data_dir, 'full_shape_val_data.pkl'), 'rb') as f:
                    temp_data = pkl.load(f)
        for index, info in enumerate(temp_data):
            if self.partial:
                partial_info = info["partial"]
                for view, data_info in partial_info.items():
                    temp_info = {}
                    temp_info["shape_id"] = info["shape_id"]
                    temp_info["semantic class"] = info["semantic class"]
                    # print("temp_info[semantic class]: ",temp_info["semantic class"])
                    temp_info["affordance"] = info["affordance"]
                    temp_info["view_id"] = view
                    temp_info["data_info"] = data_info
                    self.all_data.append(temp_info)
                    if temp_info["semantic class"] not in self.classes:
                        self.classes.append(temp_info["semantic class"])
            elif self.split != 'train' and self.rotate != 'None':
                rotate_info = temp_data_rotate[index]["rotate"][self.rotate]
                full_shape_info = info["full_shape"]
                for r, r_data in rotate_info.items():
                    temp_info = {}
                    temp_info["shape_id"] = info["shape_id"]
                    temp_info["semantic class"] = info["semantic class"]
                    temp_info["affordance"] = info["affordance"]
                    temp_info["data_info"] = full_shape_info
                    temp_info["rotate_matrix"] = r_data.astype(np.float32)
                    self.all_data.append(temp_info)
                    if temp_info["semantic class"] not in self.classes:
                        self.classes.append(temp_info["semantic class"])
            else:
                temp_info = {}
                temp_info["shape_id"] = info["shape_id"]
                temp_info["semantic class"] = info["semantic class"]
                temp_info["affordance"] = info["affordance"]
                temp_info["data_info"] = info["full_shape"]
                self.all_data.append(temp_info)
                if temp_info["semantic class"] not in self.classes:
                    self.classes.append(temp_info["semantic class"])
        
        self.classes = sorted(self.classes)
        self.classes_set = OrderedSet(self.classes)

    def __getitem__(self, index):

        data_dict = self.all_data[index]
        modelid = data_dict["shape_id"]
        modelcat = data_dict["semantic class"]
        
        # print("modelcat: ",type(modelcat))
        

        data_info = data_dict["data_info"]
        model_data = data_info["coordinate"].astype(np.float32)
        labels = data_info["label"]
        for aff in self.affordance:
            temp = labels[aff].astype(np.float32).reshape(-1, 1)
            model_data = np.concatenate((model_data, temp), axis=1)

        datas = model_data[:, :3]
        targets = model_data[:, 3:]

        if self.rotate != 'None':
            if self.split == 'train':
                if self.rotate == 'so3':
                    datas = rotate_point_cloud_SO3(
                        datas[np.newaxis, :, :]).squeeze()
                elif self.rotate == 'z':
                    datas = rotate_point_cloud_y(
                        datas[np.newaxis, :, :]).squeeze()
            else:
                r_matrix = data_dict["rotate_matrix"]
                datas = (np.matmul(r_matrix, datas.T)).T

        datas, _, _ = pc_normalize(datas)
        
        class_weights = np.ndarray((1,len(list(labels.values()))), dtype=np.float32)
        # print(type(self.classes))
        class_list = (self.classes)
        class_label = torch.zeros((len(self.classes)))
        class_label[self.classes_set.index(modelcat)] = 1
        # class_weights = torch.from_numpy(class_weights).to(torch.float32)
        for i, (key, value) in enumerate(labels.items()):
            if np.sum(value) == 0:
                class_weights[0][i] = 0.1  # 标签中没有正样本，权重为0.1
            else:
                # class_weights[0][i] = 1.0 / np.sum(value)
                class_weights[0][i] = 1.0
                
        # print(class_weights)
        
        return datas, datas, targets, modelid, modelcat, class_weights, class_list, class_label

    def __len__(self):
        return len(self.all_data)


class AffordNetDataset_Unlabel(Dataset):
    def __init__(self, data_dir):
        super().__init__()
        self.data_dir = data_dir
        self.load_data()
        self.affordance = self.all_data[0]["affordance"]
        return

    def load_data(self):
        self.all_data = []
        with open(opj(self.data_dir, 'semi_unlabel_1.pkl'), 'rb') as f:
            temp_data = pkl.load(f)
        for info in temp_data:
            temp_info = {}
            temp_info["shape_id"] = info["shape_id"]
            temp_info["semantic class"] = info["semantic class"]
            temp_info["affordance"] = info["affordance"]
            temp_info["data_info"] = info["full_shape"]
            self.all_data.append(temp_info)

    def __getitem__(self, index):
        data_dict = self.all_data[index]
        modelid = data_dict["shape_id"]
        modelcat = data_dict["semantic class"]

        data_info = data_dict["data_info"]
        datas = data_info["coordinate"].astype(np.float32)

        datas, _, _ = pc_normalize(datas)

        return datas, datas, modelid, modelcat

    def __len__(self):
        return len(self.all_data)
