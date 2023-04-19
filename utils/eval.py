# Copyright (c) Gorilla-Lab. All rights reserved.
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
from os.path import join as opj
from tqdm import tqdm
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


def get_colors(num_aff):
    HSV_tuples = [(x*1.0/num_aff, 0.5, 0.5) for x in range(num_aff)]
    RGB_tuples = map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples)
    return list(RGB_tuples)

def get_points_renderer(
    image_size=512, device=None, radius=0.01, background_color=(1, 1, 1)
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
    # print("len(affordance): ",len(affordance))
    modelids = []
    modelcats = []
    color_list = torch.Tensor(get_colors(len(affordance)))
    # print(color_list)
    color_list = color_list.repeat(2048,1,1).to(device)
    print("color_list.shape: ",color_list.shape)
    point_renderer = get_points_renderer(device=device)
    
    with torch.no_grad():
        model.eval()
        total_L2distance = 0
        count = 0.0
        for i,  temp_data in tqdm(enumerate(test_loader), total=len(test_loader), smoothing=0.9):

            (data, data1, label, modelid, modelcat, class_weights) = temp_data
            data, label = data.float().cuda(), label.float().cuda()
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            num_point = data.size()[2]
            count += batch_size * num_point
            afford_pred = torch.sigmoid(model(data))
            afford_pred = afford_pred.permute(0, 2, 1).contiguous()
            
            afford_pred_single = torch.argmax(afford_pred , dim=-1).squeeze(0)
            label_pred_single = torch.argmax(label , dim=-1).squeeze(0)
            row_idx = torch.arange(color_list.size(0)).repeat_interleave(1)
            
            afford_colors = color_list[row_idx , afford_pred_single,:].squeeze(dim=1).unsqueeze(dim=0)
            label_colors = color_list[row_idx , label_pred_single,:].squeeze(dim=1).unsqueeze(dim=0)
            point_data = data.permute(0,2,1)
            
            point_cloud_pred = Pointclouds(points = point_data, features=afford_colors)
            point_cloud_label = Pointclouds(points = point_data, features=label_colors)
            
            num_views : int = 90
            R, T = pytorch3d.renderer.look_at_view_transform(dist= 12 , elev= 0 ,azim= np.linspace(-180,180 , num_views, endpoint=False))
            cameras = pytorch3d.renderer.FoVPerspectiveCameras(device=device, fov = 60, R=R, T=T)
            lights = pytorch3d.renderer.PointLights(location = [[0,10,0]] , device=device)
            pred_point_image = point_renderer(point_cloud_pred.extend(num_views), cameras=cameras , lights=lights)
            label_point_image = point_renderer(point_cloud_label.extend(num_views), cameras=cameras , lights=lights)

            imageio.mimsave('/home/daman/AffordanceNet_daman/image/pn2/{}_{}_pred.gif'.format(modelcat[0], modelid[0]), pred_point_image.cpu().numpy(), fps=30)
            imageio.mimsave('/home/daman/AffordanceNet_daman/image/pn2/{}_{}_label.gif'.format(modelcat[0], modelid[0]), label_point_image.cpu().numpy(), fps=30)
            
            
            
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
