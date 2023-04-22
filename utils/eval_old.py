# Copyright (c) Gorilla-Lab. All rights reserved.
import torch
import pytorch3d.structures
import numpy as np
import colorsys
from os.path import join as opj
from tqdm import tqdm
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

##CLIP Import##
import clip

def get_colors(num_aff):
    HSV_tuples = [(x*1.0/num_aff, 0.5, 0.5) for x in range(num_aff)]
    RGB_tuples = map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples)
    return list(RGB_tuples)
    
def evaluation(logger, cfg, model, test_loader, affordance):
    exp_name = cfg.work_dir.split('/')[-1]
    results = torch.zeros(
        (len(test_loader), 2048, len(affordance)))
    targets = torch.zeros(
        (len(test_loader), 2048, len(affordance)))
    coordinate = np.zeros((0, 2048, 3))
    # print("len(affordance): ",len(affordance))
    modelids = []
    modelcats = []
    color_list = get_colors(len(affordance))
    with torch.no_grad():
        model.eval()
        total_L2distance = 0
        count = 0.0
        
        ##CLIP Embedding stuff##
        clip_model, preprocess = clip.load("ViT-B/32", device='cuda')
        
        affordances_list = ['grasp','contain','lift','openable','layable','sittable',
                            'support','wrap grasp', 'pourable', 'move', 'display', 
                            'pushable', 'pull', 'listen', 'wear', 'press', 'cut', 'stab']
        
        cos = torch.nn.CosineSimilarity(dim=-1,eps=1e-6)
        ##CLIP Embedding Stuff ends##

        for i,  temp_data in tqdm(enumerate(test_loader), total=len(test_loader), smoothing=0.9):
            (data, data1, label, modelid, modelcat, class_weights) = temp_data


            affordance_token = clip.tokenize(affordances_list).cuda()
            affordance_embeddings = clip_model.encode_text(affordance_token)
            affordance_embeddings = affordance_embeddings.repeat(data.shape[0],data.shape[1],1,1) # M(18) x 512 shape


            data, label = data.float().cuda(), label.float().cuda()
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            num_point = data.size()[2]
            count += batch_size * num_point
            # afford_pred = torch.sigmoid(model(data))
            # afford_pred = afford_pred.permute(0, 2, 1).contiguous()
            afford_clip_emb = model(data)
            afford_pred_corr = cos(afford_clip_emb,affordance_embeddings)
            # print(afford_pred_corr.shape)
            # print(torch.amin(afford_pred_corr,dim=-1).shape)
            afford_pred_corr = afford_pred_corr - torch.amin(afford_pred_corr,dim=-1).unsqueeze(-1)
            afford_pred = afford_pred_corr/torch.amax(afford_pred_corr,dim=-1).unsqueeze(-1)

            # afford_colors 
            # point_cloud = pytorch3d.structures.PointClouds(points = data)
            
            
            # print("afford_pred.shape: ",afford_pred.shape)
            L2distance = torch.sum(
                torch.pow(label-afford_pred, 2), dim=(0, 1))
            total_L2distance += L2distance
            score = afford_pred.squeeze()
            target_score = label.squeeze()
            results[i, :, :] = score
            targets[i, :, :] = target_score
            modelids.append(modelid[0])
            modelcats.append(modelcat[0])

    results = results.detach().cpu().numpy()
    targets = targets.detach().cpu().numpy()
    MSE = torch.sum(total_L2distance*1.0/count).item()
    AP = np.zeros((targets.shape[0], targets.shape[2]))
    F1 = np.zeros((targets.shape[0], targets.shape[2]))
    AUC = np.zeros((targets.shape[0], targets.shape[2]))
    IOU = np.zeros((targets.shape[0], targets.shape[2]))
    targets = targets >= 0.5
    targets = targets.astype(int)
    IOU_thres = np.linspace(0, 1, 20)
    for i in range(AP.shape[0]):
        t = targets[i, :, :]
        p = results[i, :, :]
        for j in range(t.shape[1]):
            t_true = t[:, j]
            p_score = p[:, j]
            if np.sum(t_true) == 0:
                F1[i, j] = np.nan
                AP[i, j] = np.nan
                AUC[i, j] = np.nan
                IOU[i, j] = np.nan
            else:
                ap = average_precision_score(t_true, p_score)
                AP[i, j] = ap
                p_mask = (p_score > 0.5).astype(int)
                f1 = f1_score(t_true, p_mask)
                F1[i, j] = f1
                auc = roc_auc_score(t_true, p_score)
                AUC[i, j] = auc
                temp_iou = []
                for thre in IOU_thres:
                    p_mask = (p_score >= thre).astype(int)
                    intersect = np.sum(p_mask & t_true)
                    union = np.sum(p_mask | t_true)
                    temp_iou.append(1.*intersect/union)
                temp_iou = np.array(temp_iou)
                aiou = np.mean(temp_iou)
                IOU[i, j] = aiou
    AP = np.nanmean(AP, axis=0)
    F1 = np.nanmean(F1, axis=0)
    AUC = np.nanmean(AUC, axis=0)
    IOU = np.nanmean(IOU, axis=0)
    for i in range(AP.size):
        outstr = affordance[i]+'_AP = ' + str(AP[i])
        logger.cprint(outstr)
        outstr = affordance[i]+'_AUC = ' + str(AUC[i])
        logger.cprint(outstr)
        outstr = affordance[i]+'_aIOU = ' + str(IOU[i])
        logger.cprint(outstr)
        outstr = affordance[i]+'_MSE = ' + \
            str(total_L2distance[i].item()/count)
        logger.cprint(outstr)
    outstr = 'Test :: test mAP: %.6f, test mAUC: %.6f, test maIOU: %.6f, test MSE: %.6f' % (
        np.mean(AP), np.mean(AUC), np.mean(IOU), MSE)
    logger.cprint(outstr)
    return np.mean(AP)
