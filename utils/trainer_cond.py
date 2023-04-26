# Copyright (c) Gorilla-Lab. All rights reserved.
import numpy as np
import torch
import torch.nn as nn
import os
from os.path import join as opj
from tqdm import tqdm
from tensorboardX import SummaryWriter
from utils import *
from utils.eval import evaluation
from time import time
import wandb

##CLIP Imports
import clip
import random

# wandb.login()
class Trainer(object):
    def __init__(self, cfg, running):
        super().__init__()
        self.cfg = cfg
        self.work_dir = self.cfg.work_dir
        self.writer = SummaryWriter(self.work_dir)
        self.logger = running['logger']
        self.model = running["model"]
        self.dataset_dict = running["dataset_dict"]
        self.loader_dict = running["loader_dict"]
        self.train_loader = self.loader_dict.get("train_loader", None)
        self.val_loader = self.loader_dict.get("val_loader", None)
        self.train_unlabel_loader = self.loader_dict.get(
            "train_unlabel_loader", None)
        if self.train_unlabel_loader is not None:
            self.unlabel_loader_iter = iter(self.train_unlabel_loader)
        self.test_loader = self.loader_dict.get("test_loader", None)
        self.loss = running["loss"]
        self.optim_dict = running["optim_dict"]
        self.optimizer = self.optim_dict.get("optimizer", None)
        self.scheduler = self.optim_dict.get("scheduler", None)
        self.epoch = 0
        self.best_val_iou = 0.0
        self.bn_momentum = self.cfg.get('bn_momentum', None)
        self.affordance = cfg.data.category
        return

    def train(self):
        train_loss = 0.0
        loss_label = 'weighted'
        count = 0.0
        self.model.train()
        num_batches = len(self.train_loader)
        start = time()
        self.logger.cprint("Epoch(%d) begin training........" % self.epoch)
        
        
        ##CLIP Embedding stuff##
        clip_model, preprocess = clip.load("ViT-B/32", device='cuda')
        
        affordances_list =['grasp', 'contain', 'lift', 'openable', 'layable', 'sittable',
              'support', 'wrap_grasp', 'pourable', 'move', 'displaY', 'pushable', 'pull',
              'listen', 'wear', 'press', 'cut', 'stab']

        # affordances_list = ['cut','stab']
        
        cos = torch.nn.CosineSimilarity(dim=-1,eps=1e-6)
        i = 0
        ##CLIP Embedding Stuff ends##
        wandb.init(project=self.cfg.model.type, config=self.cfg.training_cfg)
        for data, data1, label, _, _, class_weights in tqdm(self.train_loader, total=len(self.train_loader), smoothing=0.9):

            data, label, class_weights = data.float().cuda(), label.float().cuda(), class_weights.float().cuda()
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            num_point = data.size()[2]
            
            for j in range(1):
                label_idx = random.randint(0,17)
                # print("shapes",data.shape,label.shape)
                affordance_token = clip.tokenize(affordances_list[label_idx]).cuda()
                affordance_embeddings = clip_model.encode_text(affordance_token)#.unsqueeze(0)
                # affordance_embeddings = affordance_embeddings.repeat(data.shape[0]s,data.shape[2],1,1) # M(18) x 512 shape

                # print(affordance_embeddings.shape)

                if self.train_unlabel_loader is not None:
                    try:
                        ul_data, ul_data1, _, _ = next(self.unlabel_loader_iter)
                        ul_data, ul_data1 = ul_data.float(), ul_data1.float()
                    except StopIteration:
                        self.unlabel_loader_iter = iter(self.train_unlabel_loader)
                        ul_data, ul_data1, _, _ = next(self.unlabel_loader_iter)
                        ul_data, ul_data1 = ul_data.float(), ul_data1.float()

                self.optimizer.zero_grad()
                if self.train_unlabel_loader is not None:
                    ul_data = ul_data.cuda().float().permute(0, 2, 1)
                    data_ = torch.cat((data, ul_data), dim=0)  # VAT
                    afford_pred = torch.sigmoid(self.model(data_))
                else:
                    # Model output here
                    # print(data.shape)
                    afford_output = (self.model(data,torch.tanh(affordance_embeddings)))

                if self.train_unlabel_loader is not None and loss_label is not 'weighted':
                    l_pred = afford_pred[:batch_size, :, :]  # VAT
                    ul_pred = afford_pred[batch_size:, :, :]  # VAT
                    loss = self.loss(self.model, data, ul_data,
                                    l_pred, label, ul_pred, self.epoch)  # VAT
                elif self.train_unlabel_loader is None and loss_label is not 'weighted':
                    loss = self.loss(afford_pred, label)
                else:
                    # loss_fn = torch.nn.MSELoss()
                    # curr_label = torch.where(label[:,:,label_idx]>0,1,0)
                    # loss_clip_emb = loss_fn(afford_output,label[:,:,label_idx])
                    # afford_output = afford_output.squeeze(1).unsqueeze(-1)
                    # curr_label = label[:,:,11].unsqueeze(-1)
                    # print(afford_output.shape,label.shape)

                    # afford_output = afford_output.unsqueeze(-1).repeat(1,1,1,18).permute(0,2,3,1)
                    # print(afford_output.shape,affordance_embeddings.shape)
                    # afford_pred_corr = (cos((afford_output),(affordance_embeddings)))

                    # softmax_fn = torch.nn.Softmax(dim=-1)
                    # afford_pred_corr = softmax_fn(afford_pred_corr)
                    # label = softmax_fn(label)
                    # print(afford_output.shape,label.shape)
                    loss = self.loss(afford_output,label[:,:,label_idx],class_weights)
                    # loss_fn = torch.nn.MSELoss()
                    # loss = loss_fn(afford_pred_corr,label)
                    # loss = loss_clip_emb 
                # print("shapes: ",afford_pred.shape,label.shape,class_weights.shape)
                if i%10 == 0:
                    print(loss)
                if i%800 ==0:
                    print(i)
                    torch.save(self.model.state_dict(), opj(self.work_dir, 'model_%d.t7' % i))

                i=i+1
                loss.backward()
                self.optimizer.step()

                count += batch_size * num_point
                train_loss += loss.item()
                wandb.log({'loss':loss.cpu().item() , 'lr':float(self.optimizer.param_groups[0]['lr'])})

        self.scheduler.step()
        if self.bn_momentum != None:
            self.model.apply(lambda x: self.bn_momentum(x, self.epoch))
        epoch_time = time() - start
        outstr = 'Train(%d), loss: %.6f, time: %d s' % (
            self.epoch, train_loss*1.0/num_batches, epoch_time//1)
        self.writer.add_scalar('Loss', train_loss*1.0/num_batches, self.epoch)
        # 获取梯度信息
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                self.writer.add_scalar('grad/{}'.format(name), torch.norm(param.grad), self.epoch)
        self.logger.cprint(outstr)
        self.epoch += 1

    def val(self):
        self.logger.cprint('Epoch(%d) begin validating......' % (self.epoch-1))
        mAP = evaluation(self.logger, self.cfg, self.model,
                         self.val_loader, self.affordance)

        if mAP >= self.best_val_iou:
            self.best_val_iou = mAP
            torch.save(self.model.state_dict(),
                       opj(self.work_dir, 'model_%d.t7' % self.epoch))
            self.logger.cprint('Saving model......')
            self.logger.cprint('Best mAP: %f' % self.best_val_iou)
        torch.save(self.model.state_dict(),
                   opj(self.work_dir, 'model.t7'))

    def test(self):
        self.logger.cprint('Begin testing......')
        evaluation(self.logger, self.cfg, self.model,
                   self.test_loader, self.affordance)
        return

    def run(self):
        EPOCH = self.cfg.training_cfg.epoch
        workflow = self.cfg.training_cfg.workflow
        if self.test_loader != None:
            epoch_runner = getattr(self, 'test')
            epoch_runner()
        else:
            while self.epoch < EPOCH:
                for key, running_epoch in workflow.items():
                    epoch_runner = getattr(self, key)
                    for e in range(running_epoch):
                        epoch_runner()
