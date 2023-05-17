# Copyright (c) Gorilla-Lab. All rights reserved.
import torch
import numpy as np
from os.path import join as opj
from tqdm import tqdm
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

import wandb
import torch
import imageio
import pytorch3d
from pytorch3d.renderer import (
    AlphaCompositor,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    PointsRasterizationSettings,
    PointsRenderer,
    PointsRasterizer,
    HardPhongShader,
)
from pytorch3d.structures import Pointclouds
import numpy as np
import colorsys

import clip

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

def get_points_renderer(
    image_size=512, device=None, radius=0.01, background_color=(1, 180/255, 18/255)
):

    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")
    raster_settings = PointsRasterizationSettings(image_size=image_size, radius=radius,)
    renderer = PointsRenderer(
        rasterizer=PointsRasterizer(raster_settings=raster_settings),
        compositor=AlphaCompositor(background_color=background_color),
    )
    return renderer

def evaluation(logger, cfg, model, test_loader, affordance):
    exp_name = cfg.work_dir.split('/')[-1]
    results = torch.zeros(
        (len(test_loader), 2048, len(affordance)))
    targets = torch.zeros(
        (len(test_loader), 2048, len(affordance)))
    coordinate = np.zeros((0, 2048, 3))
    modelids = []
    modelcats = []

    point_renderer = get_points_renderer(device=device)


    ##CLIP Embedding stuff##
    clip_model, preprocess = clip.load("ViT-B/32", device='cuda')
    
    affordances_list = ['grasp','contain','lift','openable','layable','sittable',
                        'support','wrap grasp', 'pourable', 'move', 'display', 
                        'pushable', 'pull', 'listen', 'wear', 'press', 'cut', 'stab']
    
    cos = torch.nn.CosineSimilarity(dim=-1,eps=1e-6)
    ##CLIP Embedding Stuff ends##

    with torch.no_grad():
        model.eval()
        total_L2distance = 0
        count = 0.0
        for i,  temp_data in tqdm(enumerate(test_loader), total=len(test_loader), smoothing=0.9):
            if i%150!=0:
                continue
            # if i<150:
            #     continue                
            (data, data1, label, modelid, modelcat, class_weights) = temp_data

            data, label = data.float().cuda(), label.float().cuda()
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            num_point = data.size()[2]
            count += batch_size * num_point

            affordance_token = clip.tokenize(affordances_list).cuda()
            affordance_embeddings = clip_model.encode_text(affordance_token)#.unsqueeze(0)
            affordance_embeddings = affordance_embeddings.repeat(data.shape[0],data.shape[2],1,1) # M(18) x 512 shape

            afford_output = (model(data,torch.tanh(affordance_embeddings)))
            afford_output = afford_output.unsqueeze(-1).repeat(1,1,1,18).permute(0,2,3,1)
            # print(afford_output.shape,affordance_embeddings.shape)
            afford_pred = (cos((afford_output),(affordance_embeddings)))

            # afford_pred = torch.sigmoid(model(data))
            # afford_pred = afford_pred.permute(0, 2, 1).contiguous()

            # print(afford_pred.shape)

            for l_idx in range(18):

                afford_colors = torch.ones((1,2048,3)).cuda()*0.75
                label_colors = torch.ones((1,2048,3)).cuda()

                afford_pred = torch.where(afford_pred>0.18,1,0)

                label_colors = label_colors*(label[:,:,l_idx].squeeze(0).unsqueeze(-1).repeat(1,1,3)[0,0:,:])#*10
                
                # print(label[:,:,l_idx])

                # torch.where(label_colors[:,:,1]==0 , 1 ,label_colors[:,:,1])
                # torch.where(label_colors[:,:,2:]==0 , 0 ,label_colors[:,:,2:])
                
                afford_colors = afford_colors*(afford_pred.squeeze(0)[:,l_idx].unsqueeze(-1).repeat(1,1,3)[0,0:,:])#*100            

                # torch.where(afford_colors[:,:,1]==0 , 1 ,afford_colors[:,:,1])
                # torch.where(afford_colors[:,:,2:]==0 , 0 , afford_colors[:,:,2:])

                point_data = data.permute(0,2,1)

                point_cloud_pred = Pointclouds(points = point_data, features=afford_colors)
                point_cloud_label = Pointclouds(points = point_data, features=label_colors)

                num_views : int = 90
                R, T = pytorch3d.renderer.look_at_view_transform(dist= 6 , elev= 0 ,azim= np.linspace(-180,180 , num_views, endpoint=False))
                cameras = pytorch3d.renderer.FoVPerspectiveCameras(device=device, fov = 60, R=R, T=T)
                lights = pytorch3d.renderer.PointLights(location = [[0,10,0]] , device=device)
                pred_point_image = point_renderer(point_cloud_pred.extend(num_views), cameras=cameras , lights=lights)
                label_point_image = point_renderer(point_cloud_label.extend(num_views), cameras=cameras , lights=lights)

                imageio.mimsave('/home/daman/AffordanceNet_daman/image/pn2/train/{}_{}_affordance_{}_pred.gif'.format(modelcat[0], modelid[0],l_idx), pred_point_image.cpu().numpy(), fps=30)
                imageio.mimsave('/home/daman/AffordanceNet_daman/image/pn2/train/{}_{}_affordance_{}_label.gif'.format(modelcat[0], modelid[0],l_idx), label_point_image.cpu().numpy(), fps=30)

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
        
        # wandb.log({str(affordance[i]+'_AP'):AP[i],
        #           str(affordance[i]+'_AUC'):AUC[i],
        #           str(affordance[i]+'_aIOU'):IOU[i],
        #           str(affordance[i]+'_MSE'):total_L2distance[i].item()/count})
    outstr = 'Test :: test mAP: %.6f, test mAUC: %.6f, test maIOU: %.6f, test MSE: %.6f' % (
        np.nanmean(AP), np.nanmean(AUC), np.nanmean(IOU), MSE)
    
    # wandb.log({'test mean_AP' : np.nanmean(AP),
    #            'test mean_AUC' : np.nanmean(AUC),
    #            'test mean_aIOU' : np.nanmean(IOU),
    #            'test MSE' : MSE})
    logger.cprint(outstr)
    return np.nanmean(AP)
