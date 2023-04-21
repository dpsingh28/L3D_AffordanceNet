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
        
        affordances_list = ['grasp','contain','lift','openable','layable','sittable',
                            'support','wrap grasp', 'pourable', 'move', 'display', 
                            'pushable', 'pull', 'listen', 'wear', 'press', 'cut', 'stab']
        
        # affordances_list = ['cut','stab']
        
        cos = torch.nn.CosineSimilarity(dim=-1,eps=1e-6)
        i = 0
        ##CLIP Embedding Stuff ends##
        wandb.init(project=self.cfg.model.type, config=self.cfg.training_cfg)
        for data, data1, label, _, _, class_weights in tqdm(self.train_loader, total=len(self.train_loader), smoothing=0.9):

            affordance_token = clip.tokenize(affordances_list).cuda()
            affordance_embeddings = clip_model.encode_text(affordance_token)
            affordance_embeddings = affordance_embeddings.repeat(data.shape[0],data.shape[1],1,1) # M(18) x 512 shape
            # affordance_embeddings = affordance_embeddings.permute(0,1,2,3)
            # print("affordance embeddings",affordance_embeddings.shape)

            if self.train_unlabel_loader is not None:
                try:
                    ul_data, ul_data1, _, _ = next(self.unlabel_loader_iter)
                    ul_data, ul_data1 = ul_data.float(), ul_data1.float()
                except StopIteration:
                    self.unlabel_loader_iter = iter(self.train_unlabel_loader)
                    ul_data, ul_data1, _, _ = next(self.unlabel_loader_iter)
                    ul_data, ul_data1 = ul_data.float(), ul_data1.float()

            data, label, class_weights = data.float().cuda(), label.float().cuda(), class_weights.float().cuda()
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            num_point = data.size()[2]
            self.optimizer.zero_grad()
            if self.train_unlabel_loader is not None:
                ul_data = ul_data.cuda().float().permute(0, 2, 1)
                data_ = torch.cat((data, ul_data), dim=0)  # VAT
                afford_pred = torch.sigmoid(self.model(data_))
            else:
                # model_out,afford_clip_emb = self.model(data)
                afford_clip_emb = self.model(data)
                afford_clip_emb = torch.tanh(afford_clip_emb)
                # afford_pred = torch.sigmoid(model_out)
            # afford_pred = afford_pred.permute(0, 2, 1).contiguous()
            if self.train_unlabel_loader is not None and loss_label is not 'weighted':
                l_pred = afford_pred[:batch_size, :, :]  # VAT
                ul_pred = afford_pred[batch_size:, :, :]  # VAT
                loss = self.loss(self.model, data, ul_data,
                                 l_pred, label, ul_pred, self.epoch)  # VAT
            elif self.train_unlabel_loader is None and loss_label is not 'weighted':
                # afford_pred_corr = cos(afford_pred_corr,affordance_embeddings)
                # print("afford_pred_corr",afford_pred_corr.shape)
                # loss = self.loss(afford_pred_corr, label)
                loss = self.loss(afford_pred, label)
            else:
                # print(torch.max(affordance_embeddings),torch.min(affordance_embeddings))
                afford_pred_corr = (cos(afford_clip_emb,torch.tanh(affordance_embeddings)))
                # print(afford_pred_corr.shape)
                # print(torch.amin(afford_pred_corr,dim=-1).shape)

                # label = torch.where(label>0.2,1.0,0.0)

                # softmax_fn = torch.nn.Softmax(dim=-1)
                # afford_pred_corr = softmax_fn(afford_pred_corr)
                # label = softmax_fn(label)
                # print(torch.max(afford_pred_corr),torch.min(afford_pred_corr),torch.max(label),torch.min(label))
                # print("afford_pred_corr",afford_pred_corr.shape)
                # print(afford_pred_corr.shape,label[:,:,16:].shape,class_weights.shape)
                loss_fn = torch.nn.MSELoss()
                loss_clip_emb = loss_fn(afford_pred_corr,label[:,:,:])
                # loss_clip_emb = self.loss(afford_pred_corr, label[:,:,16:], class_weights[:,:,16:])
                # print(torch.sum(label[0,:,:],dim=0))
                # loss = torch.mean(class_weights*(torch.abs(afford_pred_corr-label)))
                
                # print(afford_pred_corr[0,-2,:],label[0,-2,:])
                # loss_seg = self.loss(afford_pred, label, class_weights)
                loss = loss_clip_emb #+ loss_seg
                # print(afford_pred.shape,label.shape)
            # print("shapes: ",afford_pred.shape,label.shape,class_weights.shape)
            if i%10 == 0:
                print(loss)
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
