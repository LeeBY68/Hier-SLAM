import argparse
import os
import shutil
import sys
import time
from importlib.machinery import SourceFileLoader
import re

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.insert(0, _BASE_DIR)

print("System Paths:")
for p in sys.path:
    print(p)

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import wandb

from datasets.gradslam_datasets import (
    load_dataset_config,
    ICLDataset,
    ReplicaDataset,
    ReplicaDataset_semantic,
    ReplicaV2Dataset,
    AzureKinectDataset,
    ScannetDataset,
    ScannetDataset_semantic,
    Ai2thorDataset,
    Record3DDataset,
    RealsenseDataset,
    TUMDataset,
    ScannetPPDataset,
    NeRFCaptureDataset
)
from utils.common_utils import seed_everything, save_params_ckpt, save_params
from utils.eval_helpers import report_loss, report_loss_semantic, report_progress, report_progress_newrender, eval_newrender, eval_semantic_tree_newrender, eval_semantic_newrender, \
        transfer_tree_label, semantic_label_vis, semantic_label_vis_replica, eval_semantic_single, transfer_tree_2_label
from utils.keyframe_selection import keyframe_selection_overlap
from utils.recon_helpers import setup_camera
from utils.slam_helpers import (
    transformed_params2rendervar, transformed_params2depthplussilhouette, transformed_params2rendervar_semantic,
    transform_to_frame, l1_loss_v1, matrix_to_quaternion
)
from utils.slam_external import calc_ssim, build_rotation, prune_gaussians, densify
from utils.graphics_utils import RGB2SH, SH2RGB

from diff_gaussian_rasterization import GaussianRasterizer as Renderer
from diff_gaussian_rasterization import GaussianRasterizer_semantic as Renderer_semantic


def update_poses(params, sliding_window_kf):
    """
    update poses in sliding_window_kf['est_w2c'] 
    """
    with torch.no_grad():
        for i, kf_data in enumerate(sliding_window_kf):
            # Get the current estimated rotation & translation
            curr_cam_rot = F.normalize(params['cam_unnorm_rots'][..., kf_data['id']].detach())
            curr_cam_tran = params['cam_trans'][..., kf_data['id']].detach()
            curr_w2c = torch.eye(4).cuda().float()
            curr_w2c[:3, :3] = build_rotation(curr_cam_rot)
            curr_w2c[:3, 3] = curr_cam_tran

            kf_data['est_w2c'] = curr_w2c
        
# tools & visualization
def transfer_label_2_softmax(label_map, num_semantic):
    """
    label_map: [level_num, h, w]
    """
    for i_level in range(len(num_semantic)-1):
        num_classes_level = num_semantic[i_level]
        label_map_level = label_map[i_level, :, :]      # [h, w]
        one_hot = torch.zeros(num_classes_level, label_map_level.shape[0], label_map_level.shape[1])
        one_hot = one_hot.scatter_(0, label_map_level.unsqueeze(0).long(), 1)
        softmax = F.softmax(one_hot, dim=0)
        if i_level == 0:
            label_map_softmax = softmax
        else:
            label_map_softmax = torch.cat([label_map_softmax, softmax], dim=0)

    return label_map_softmax

# loss calculation
def transfer_tree_rendered_labelmap(rendered_map, i_level, dataset):
    """
    transfer one-hot tree representation -> each level label map
    rendered_map: [num_treeall_classes, h, w]
    i_level: int, the level index
    dataset: Dataset
    im_semantic_cal: [h*w, num_classes_level]
    """

    num_classes_level = dataset.num_semantic[i_level]
    if i_level == 0:
        idx_begin = 0
    else:
        idx_begin = sum(dataset.num_semantic[:i_level])
    idx_end = idx_begin + num_classes_level

    rendered_map_level = rendered_map[idx_begin:idx_end, :, :]
    rendered_map_cal = rendered_map_level.permute(1,2,0)                            # [num_classes, h, w] -> [h, w, num_classes]
    rendered_map_cal = rendered_map_cal.view(-1, rendered_map_cal.size(2)) 

    return rendered_map_cal


def get_dataset(config_dict, basedir, sequence, **kwargs):
    if config_dict["dataset_name"].lower() in ["icl"]:
        return ICLDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["replica"]:
        return ReplicaDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["replica_semantic"]:
        return ReplicaDataset_semantic(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["replicav2"]:
        return ReplicaV2Dataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["azure", "azurekinect"]:
        return AzureKinectDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["scannet"]:
        return ScannetDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["scannet_semantic"]:
        return ScannetDataset_semantic(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["ai2thor"]:
        return Ai2thorDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["record3d"]:
        return Record3DDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["realsense"]:
        return RealsenseDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["tum"]:
        return TUMDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["scannetpp"]:
        return ScannetPPDataset(basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["nerfcapture"]:
        return NeRFCaptureDataset(basedir, sequence, **kwargs)
    else:
        raise ValueError(f"Unknown dataset name {config_dict['dataset_name']}")

def get_pointcloud(color, depth, intrinsics, w2c, transform_pts=True, 
                   mask=None, compute_mean_sq_dist=False, mean_sq_dist_method="projective"):
    width, height = color.shape[2], color.shape[1]
    CX = intrinsics[0][2]
    CY = intrinsics[1][2]
    FX = intrinsics[0][0]
    FY = intrinsics[1][1]

    # Compute indices of pixels
    x_grid, y_grid = torch.meshgrid(torch.arange(width).cuda().float(), 
                                    torch.arange(height).cuda().float(),
                                    indexing='xy')
    xx = (x_grid - CX)/FX
    yy = (y_grid - CY)/FY
    xx = xx.reshape(-1)
    yy = yy.reshape(-1)
    depth_z = depth[0].reshape(-1)

    # Initialize point cloud
    pts_cam = torch.stack((xx * depth_z, yy * depth_z, depth_z), dim=-1)
    if transform_pts:
        pix_ones = torch.ones(height * width, 1).cuda().float()
        pts4 = torch.cat((pts_cam, pix_ones), dim=1)
        c2w = torch.inverse(w2c)
        pts = (c2w @ pts4.T).T[:, :3]
    else:
        pts = pts_cam

    # Compute mean squared distance for initializing the scale of the Gaussians
    if compute_mean_sq_dist:
        if mean_sq_dist_method == "projective":
            # Projective Geometry (this is fast, farther -> larger radius)
            scale_gaussian = depth_z / ((FX + FY)/2)
            mean3_sq_dist = scale_gaussian**2
        else:
            raise ValueError(f"Unknown mean_sq_dist_method {mean_sq_dist_method}")
    
    # Colorize point cloud
    cols = torch.permute(color, (1, 2, 0)).reshape(-1, 3) # (C, H, W) -> (H, W, C) -> (H * W, C)
    point_cld = torch.cat((pts, cols), -1)

    # Select points based on mask
    if mask is not None:
        point_cld = point_cld[mask]
        if compute_mean_sq_dist:
            mean3_sq_dist = mean3_sq_dist[mask]

    if compute_mean_sq_dist:
        return point_cld, mean3_sq_dist
    else:
        return point_cld

def get_pointcloud_semantic(color, depth, label_gt, intrinsics, w2c, num_semantic, transform_pts=True, 
                   mask=None, compute_mean_sq_dist=False, mean_sq_dist_method="projective"):
    """
    color: [3, h, w]
    depth: [1, h, w]
    label_gt: [1, h, w]
    """

    # label_gt [1,h,w] -> label_map [num_classes,h,w]
    label_map, num_classes = label2map(label_gt, num_semantic)

    width, height = color.shape[2], color.shape[1]
    CX = intrinsics[0][2]
    CY = intrinsics[1][2]
    FX = intrinsics[0][0]
    FY = intrinsics[1][1]

    # Compute indices of pixels
    x_grid, y_grid = torch.meshgrid(torch.arange(width).cuda().float(), 
                                    torch.arange(height).cuda().float(),
                                    indexing='xy')
    xx = (x_grid - CX)/FX
    yy = (y_grid - CY)/FY
    xx = xx.reshape(-1)
    yy = yy.reshape(-1)
    depth_z = depth[0].reshape(-1)

    # Initialize point cloud
    pts_cam = torch.stack((xx * depth_z, yy * depth_z, depth_z), dim=-1)
    if transform_pts:
        pix_ones = torch.ones(height * width, 1).cuda().float()
        pts4 = torch.cat((pts_cam, pix_ones), dim=1)
        c2w = torch.inverse(w2c)
        pts = (c2w @ pts4.T).T[:, :3]
    else:
        pts = pts_cam

    # Compute mean squared distance for initializing the scale of the Gaussians
    if compute_mean_sq_dist:
        if mean_sq_dist_method == "projective":
            # Projective Geometry (this is fast, farther -> larger radius)
            scale_gaussian = depth_z / ((FX + FY)/2)
            mean3_sq_dist = scale_gaussian**2                                   # [H * W]
        else:
            raise ValueError(f"Unknown mean_sq_dist_method {mean_sq_dist_method}")
    
    # Colorize point cloud
    cols = torch.permute(color, (1, 2, 0)).reshape(-1, 3) # (C, H, W) -> (H, W, C) -> (H * W, C)   [H * W, 3]
    point_cld = torch.cat((pts, cols), -1)                # [H * W, 6]
    point_sem = torch.permute(label_map, (1, 2, 0)).reshape(-1, num_classes)

    # Select points based on mask
    if mask is not None:
        point_cld = point_cld[mask]
        point_sem = point_sem[mask]
        if compute_mean_sq_dist:
            mean3_sq_dist = mean3_sq_dist[mask]

    if compute_mean_sq_dist:
        return point_cld, mean3_sq_dist, point_sem
    else:
        return point_cld

def get_pointcloud_semantic_tree(dataset, color, depth, label_gt, intrinsics, w2c, transform_pts=True, 
                   mask=None, compute_mean_sq_dist=False, mean_sq_dist_method="projective"):
    """
    color: [3, h, w]
    depth: [1, h, w]
    label_gt: [num_levels, h, w]
    """

    # label_gt [tree_level_num, h, w] -> label_map [num_classes(26), h, w]
    label_map, num_classes = label2map_tree_new(label_gt, dataset.num_semantic)

    width, height = color.shape[2], color.shape[1]
    CX = intrinsics[0][2]
    CY = intrinsics[1][2]
    FX = intrinsics[0][0]
    FY = intrinsics[1][1]

    # Compute indices of pixels
    x_grid, y_grid = torch.meshgrid(torch.arange(width).cuda().float(), 
                                    torch.arange(height).cuda().float(),
                                    indexing='xy')
    xx = (x_grid - CX)/FX
    yy = (y_grid - CY)/FY
    xx = xx.reshape(-1)
    yy = yy.reshape(-1)
    depth_z = depth[0].reshape(-1)

    # Initialize point cloud
    pts_cam = torch.stack((xx * depth_z, yy * depth_z, depth_z), dim=-1)
    if transform_pts:
        pix_ones = torch.ones(height * width, 1).cuda().float()
        pts4 = torch.cat((pts_cam, pix_ones), dim=1)
        c2w = torch.inverse(w2c)
        pts = (c2w @ pts4.T).T[:, :3]
    else:
        pts = pts_cam

    # Compute mean squared distance for initializing the scale of the Gaussians
    if compute_mean_sq_dist:
        if mean_sq_dist_method == "projective":
            # Projective Geometry (this is fast, farther -> larger radius)
            scale_gaussian = depth_z / ((FX + FY)/2)
            mean3_sq_dist = scale_gaussian**2                                   # [H * W]
        else:
            raise ValueError(f"Unknown mean_sq_dist_method {mean_sq_dist_method}")
    
    # Colorize point cloud
    cols = torch.permute(color, (1, 2, 0)).reshape(-1, 3) # (C, H, W) -> (H, W, C) -> (H * W, C)   [H * W, 3]
    point_cld = torch.cat((pts, cols), -1)                # [H * W, 6]
    point_sem = torch.permute(label_map, (1, 2, 0)).reshape(-1, num_classes)

    # Select points based on mask
    if mask is not None:
        point_cld = point_cld[mask]
        point_sem = point_sem[mask]
        if compute_mean_sq_dist:
            mean3_sq_dist = mean3_sq_dist[mask]

    if compute_mean_sq_dist:
        return point_cld, mean3_sq_dist, point_sem
    else:
        return point_cld

def initialize_params(init_pt_cld, num_frames, mean3_sq_dist, gaussian_distribution):
    num_pts = init_pt_cld.shape[0]
    means3D = init_pt_cld[:, :3] # [num_gaussians, 3]
    unnorm_rots = np.tile([1, 0, 0, 0], (num_pts, 1)) # [num_gaussians, 4]
    logit_opacities = torch.zeros((num_pts, 1), dtype=torch.float, device="cuda")
    if gaussian_distribution == "isotropic":
        log_scales = torch.tile(torch.log(torch.sqrt(mean3_sq_dist))[..., None], (1, 1))
    elif gaussian_distribution == "anisotropic":
        log_scales = torch.tile(torch.log(torch.sqrt(mean3_sq_dist))[..., None], (1, 3))
    else:
        raise ValueError(f"Unknown gaussian_distribution {gaussian_distribution}")
    params = {
        'means3D': means3D,
        'rgb_colors': init_pt_cld[:, 3:6],
        'unnorm_rotations': unnorm_rots,
        'logit_opacities': logit_opacities,
        'log_scales': log_scales,
    }

    # Initialize a single gaussian trajectory to model the camera poses relative to the first frame
    cam_rots = np.tile([1, 0, 0, 0], (1, 1))
    cam_rots = np.tile(cam_rots[:, :, None], (1, 1, num_frames))
    params['cam_unnorm_rots'] = cam_rots
    params['cam_trans'] = np.zeros((1, 3, num_frames))

    for k, v in params.items():
        # Check if value is already a torch tensor
        if not isinstance(v, torch.Tensor):
            params[k] = torch.nn.Parameter(torch.tensor(v).cuda().float().contiguous().requires_grad_(True))
        else:
            params[k] = torch.nn.Parameter(v.cuda().float().contiguous().requires_grad_(True))

    variables = {'max_2D_radius': torch.zeros(params['means3D'].shape[0]).cuda().float(),
                 'means2D_gradient_accum': torch.zeros(params['means3D'].shape[0]).cuda().float(),
                 'denom': torch.zeros(params['means3D'].shape[0]).cuda().float(),
                 'timestep': torch.zeros(params['means3D'].shape[0]).cuda().float()}

    return params, variables

def initialize_semantic_params(init_pt_cld, init_pt_sem, num_frames, mean3_sq_dist, num_labels):
    
    flag_init = 2   # 0: initial all 0; 1: initial randn; 2: randm within [0,1]; 3. 2d-semantic-gt
    flag_noise = False
    softmax = torch.nn.Softmax(dim=-1)
    
    num_pts = init_pt_cld.shape[0]
    means3D = init_pt_cld[:, :3]                                                    # [num_gaussians, 3]
    unnorm_rots = np.tile([1, 0, 0, 0], (num_pts, 1))                               # [num_gaussians, 4]
    logit_opacities = torch.zeros((num_pts, 1), dtype=torch.float, device="cuda")   # [num_gaussians, 1]
    if flag_init == 0:
        logit_semantic = torch.zeros((num_pts, num_labels), dtype=torch.float, device="cuda")   # [num_gaussians, num_labels]
    elif flag_init == 1:
        logit_semantic = RGB2SH(torch.rand((num_pts, num_labels), device="cuda"))
    elif flag_init == 2:
        logit_semantic = torch.rand((num_pts, num_labels), device="cuda")
    elif flag_init == 3:
        if flag_noise:
            init_pt_sem = addnoise(init_pt_sem)
        logit_semantic = softmax(init_pt_sem)

    params = {                                          # N = num_pts = num_3dgs
        'means3D': means3D,                             # [N,3]    
        'rgb_colors': init_pt_cld[:, 3:6],              # [N,3] 
        'unnorm_rotations': unnorm_rots,                # [N,4] 
        'logit_opacities': logit_opacities,             # [N,1] 
        'log_scales': torch.tile(torch.log(torch.sqrt(mean3_sq_dist))[..., None], (1, 1)),  # [N] -> [N,1]
        'semantic': logit_semantic,                     # [N, num_sem]
    }

    # Initialize a single gaussian trajectory to model the camera poses relative to the first frame
    cam_rots = np.tile([1, 0, 0, 0], (1, 1))
    cam_rots = np.tile(cam_rots[:, :, None], (1, 1, num_frames))    # [1,4,num_frames]: q
    params['cam_unnorm_rots'] = cam_rots                            # unnormed q
    params['cam_trans'] = np.zeros((1, 3, num_frames))              # translation t

    for k, v in params.items():
        # Check if value is already a torch tensor
        if not isinstance(v, torch.Tensor):
            params[k] = torch.nn.Parameter(torch.tensor(v).cuda().float().contiguous().requires_grad_(True))
        else:
            params[k] = torch.nn.Parameter(v.cuda().float().contiguous().requires_grad_(True))

    variables = {'max_2D_radius': torch.zeros(params['means3D'].shape[0]).cuda().float(),                   # [N,1]
                 'means2D_gradient_accum': torch.zeros(params['means3D'].shape[0]).cuda().float(),          # [N,1]
                 'denom': torch.zeros(params['means3D'].shape[0]).cuda().float(),                           # [N,1]
                 'timestep': torch.zeros(params['means3D'].shape[0]).cuda().float()}                        # [N,1]

    return params, variables

def initialize_optimizer(params, lrs_dict, tracking):
    lrs = lrs_dict
    param_groups = [{'params': [v], 'name': k, 'lr': lrs[k]} for k, v in params.items()]
    if tracking:
        return torch.optim.Adam(param_groups)
    else:
        return torch.optim.Adam(param_groups, lr=0.0, eps=1e-15)

def initialize_first_timestep(dataset, num_frames, scene_radius_depth_ratio, 
                              mean_sq_dist_method, densify_dataset=None, gaussian_distribution=None):
    # Get RGB-D Data & Camera Parameters
    color, depth, intrinsics, pose = dataset[0]

    # Process RGB-D Data
    color = color.permute(2, 0, 1) / 255 # (H, W, C) -> (C, H, W)
    depth = depth.permute(2, 0, 1)       # (H, W, C) -> (C, H, W)
    
    # Process Camera Parameters
    intrinsics = intrinsics[:3, :3]
    w2c = torch.linalg.inv(pose)

    # Setup Camera
    cam = setup_camera(color.shape[2], color.shape[1], intrinsics.cpu().numpy(), w2c.detach().cpu().numpy())

    if densify_dataset is not None:
        # Get Densification RGB-D Data & Camera Parameters
        color, depth, densify_intrinsics, _ = densify_dataset[0]
        color = color.permute(2, 0, 1) / 255    # (H, W, C) -> (C, H, W)
        depth = depth.permute(2, 0, 1)          # (H, W, C) -> (C, H, W)
        densify_intrinsics = densify_intrinsics[:3, :3]
        densify_cam = setup_camera(color.shape[2], color.shape[1], densify_intrinsics.cpu().numpy(), w2c.detach().cpu().numpy())
    else:
        densify_intrinsics = intrinsics

    # Get Initial Point Cloud (PyTorch CUDA Tensor)
    mask = (depth > 0) # Mask out invalid depth values
    mask = mask.reshape(-1)
    init_pt_cld, mean3_sq_dist = get_pointcloud(color, depth, densify_intrinsics, w2c, 
                                                mask=mask, compute_mean_sq_dist=True, 
                                                mean_sq_dist_method=mean_sq_dist_method)

    # Initialize Parameters
    params, variables = initialize_params(init_pt_cld, num_frames, mean3_sq_dist, gaussian_distribution)

    # Initialize an estimate of scene radius for Gaussian-Splatting Densification
    variables['scene_radius'] = torch.max(depth)/scene_radius_depth_ratio

    if densify_dataset is not None:
        return params, variables, intrinsics, w2c, cam, densify_intrinsics, densify_cam
    else:
        return params, variables, intrinsics, w2c, cam

def initialize_first_timestep_semantic(dataset, num_frames, num_semantic, scene_radius_depth_ratio, mean_sq_dist_method, densify_dataset=None):
    # Get RGB-D Data & Camera Parameters from first frame]
    color, depth, intrinsics, pose, label_gt = dataset[0]

    # Process RGB-D Data
    color = color.permute(2, 0, 1) / 255         # (H, W, C) -> (C, H, W)
    depth = depth.permute(2, 0, 1)               # (H, W, C) -> (C, H, W)
    label_gt = label_gt.permute(2, 0, 1)         # (H, W, C) -> (C, H, W)
    
    # Process Camera Parameters
    intrinsics = intrinsics[:3, :3]
    w2c = torch.linalg.inv(pose)

    # Setup Camera ---------
    cam = setup_camera(color.shape[2], color.shape[1], intrinsics.cpu().numpy(), w2c.detach().cpu().numpy())
    # Setup Camera ---------

    if densify_dataset is not None:     # (not run) skip this, need attention 
        # Get Densification RGB-D Data & Camera Parameters
        color, depth, densify_intrinsics, _ = densify_dataset[0]
        color = color.permute(2, 0, 1) / 255                        # (H, W, C) -> (C, H, W)
        depth = depth.permute(2, 0, 1)                              # (H, W, C) -> (C, H, W)
        densify_intrinsics = densify_intrinsics[:3, :3]
        densify_cam = setup_camera(color.shape[2], color.shape[1], densify_intrinsics.cpu().numpy(), w2c.detach().cpu().numpy())
    else:
        densify_intrinsics = intrinsics

    # Get Initial Point Cloud (PyTorch CUDA Tensor)
    mask = (depth > 0)                  # Mask out invalid depth values
    mask = mask.reshape(-1)

    init_pt_cld, mean3_sq_dist, init_pt_sem = get_pointcloud_semantic(color, depth, label_gt, densify_intrinsics, w2c, num_semantic,
                                                mask=mask, compute_mean_sq_dist=True, 
                                                mean_sq_dist_method=mean_sq_dist_method)
    
    # Initialize Parameters (add semantic info)
    print("[initialize_first_timestep_semantic] num_semantic is: ", num_semantic)
    params, variables = initialize_semantic_params(init_pt_cld, init_pt_sem, num_frames, mean3_sq_dist, num_semantic)
    
    # Initialize an estimate of scene radius for Gaussian-Splatting Densification
    variables['scene_radius'] = torch.max(depth)/scene_radius_depth_ratio

    if densify_dataset is not None:
        return params, variables, intrinsics, w2c, cam, densify_intrinsics, densify_cam
    else:
        return params, variables, intrinsics, w2c, cam

def initialize_first_timestep_semantic_tree(dataset, num_frames, scene_radius_depth_ratio, mean_sq_dist_method, densify_dataset=None):
    # Get RGB-D Data & Camera Parameters from first frame]
    if dataset.use_pyramid:
        color, depth, intrinsics, pose, label_gt, color_pyramid, depth_pyramid, semantic_pyramid = dataset[0]
    else:
        color, depth, intrinsics, pose, label_gt = dataset[0]
    
    # Process RGB-D Data
    color = color.permute(2, 0, 1) / 255         # (H, W, C) -> (C, H, W)
    depth = depth.permute(2, 0, 1)               # (H, W, C) -> (C, H, W)
    
    # pyramid
    if dataset.use_pyramid:
        for i in range(len(color_pyramid)):
            color_pyramid[i] = color_pyramid[i].permute(2, 0, 1) / 255
            depth_pyramid[i] = depth_pyramid[i].permute(2, 0, 1)
    
    # Process Camera Parameters
    intrinsics = intrinsics[:3, :3]
    w2c = torch.linalg.inv(pose)

    # Setup Camera ---------
    cam = setup_camera(color.shape[2], color.shape[1], intrinsics.cpu().numpy(), w2c.detach().cpu().numpy())
    if dataset.use_pyramid:
        intrinsics_use = intrinsics
        intrinsics_py = []
        intrinsics_py.append(intrinsics_use)
        cam_py = []
        cam_py.append(cam)
        for i in range(1, dataset.pyramid_level):
            intrinsics_use = intrinsics_use / 2
            intrinsics_use[2,2] = 1.0
            intrinsics_py.append(intrinsics_use)
            cam_py.append(setup_camera(color_pyramid[i].shape[2], color_pyramid[i].shape[1], intrinsics_use.cpu().numpy(), w2c.detach().cpu().numpy()))
    # Setup Camera ---------

    if densify_dataset is not None:
        # Get Densification RGB-D Data & Camera Parameters
        color, depth, densify_intrinsics, _ = densify_dataset[0]
        color = color.permute(2, 0, 1) / 255                        # (H, W, C) -> (C, H, W)
        depth = depth.permute(2, 0, 1)                              # (H, W, C) -> (C, H, W)
        densify_intrinsics = densify_intrinsics[:3, :3]
        densify_cam = setup_camera(color.shape[2], color.shape[1], densify_intrinsics.cpu().numpy(), w2c.detach().cpu().numpy())
    else:                                       
        densify_intrinsics = intrinsics

    # Get Initial Point Cloud (PyTorch CUDA Tensor)
    mask = (depth > 0)                  # Mask out invalid depth values
    mask = mask.reshape(-1)
    
    init_pt_cld, mean3_sq_dist, init_pt_sem = get_pointcloud_semantic_tree(dataset, color, depth, label_gt, densify_intrinsics, w2c, 
                                                mask=mask, compute_mean_sq_dist=True, 
                                                mean_sq_dist_method=mean_sq_dist_method)
    num_semantic = init_pt_sem.shape[1]

    # Initialize Parameters (add semantic info)
    print("[initialize_first_timestep_semantic] num_semantic is: ", num_semantic)
    params, variables = initialize_semantic_params(init_pt_cld, init_pt_sem, num_frames, mean3_sq_dist, num_semantic)

    # Initialize an estimate of scene radius for Gaussian-Splatting Densification
    variables['scene_radius'] = torch.max(depth)/scene_radius_depth_ratio

    if densify_dataset is not None:
        return params, variables, intrinsics, intrinsics_py, w2c, cam, densify_intrinsics, densify_cam
    else:
        if dataset.use_pyramid:
            return params, variables, intrinsics, intrinsics_py, w2c, cam, cam_py
        else:
            return params, variables, intrinsics, w2c, cam

# without semantic
def get_loss(params, curr_data, variables, iter_time_idx, loss_weights, use_sil_for_loss,
             sil_thres, use_l1, ignore_outlier_depth_loss, tracking=False, 
             mapping=False, do_ba=False, plot_dir=None, visualize_tracking_loss=False, tracking_iteration=None):
    # Initialize Loss Dictionary
    flag_showtime = False
    flag_printloss = False
    losses = {}

    if tracking:
        # Get current frame Gaussians, where only the camera pose gets gradient
        transformed_gaussians = transform_to_frame(params, iter_time_idx, 
                                             gaussians_grad=False,
                                             camera_grad=True)
    elif mapping:
        if do_ba:
            # Get current frame Gaussians, where both camera pose and Gaussians get gradient
            transformed_gaussians = transform_to_frame(params, iter_time_idx,
                                                 gaussians_grad=True,
                                                 camera_grad=True)
        else:
            # Get current frame Gaussians, where only the Gaussians get gradient
            transformed_gaussians = transform_to_frame(params, iter_time_idx,
                                                 gaussians_grad=True,
                                                 camera_grad=False)
    else:
        # Get current frame Gaussians, where only the Gaussians get gradient
        transformed_gaussians = transform_to_frame(params, iter_time_idx,
                                             gaussians_grad=True,
                                             camera_grad=False)

    # Initialize Render Variables
    rendervar = transformed_params2rendervar(params, transformed_gaussians)
    tracking_time_3 = time.time()
    
    # RGB Rendering
    rendervar['means2D'].retain_grad()
    im, radius, rendered_depth, rendered_median_depth, rendered_final_opcity, rendered_mask = Renderer(raster_settings=curr_data['cam'])(**rendervar)
    
    tracking_time_4 = time.time()
    if flag_showtime:
        print("time usage -> render im {}*{}*{}: {} s".format( im.shape[0], im.shape[1], im.shape[2], (tracking_time_4 - tracking_time_3)) )
    
    # ====== depth =======
    depth = rendered_depth

    # get mask
    presence_sil_mask = (rendered_final_opcity > sil_thres)
    nan_mask = (~torch.isnan(depth))
    mask = (curr_data['depth'] > 0)
    mask = mask & nan_mask
    if tracking and use_sil_for_loss:
        mask = mask & presence_sil_mask
    # =============================================

    # Depth loss
    if use_l1:
        mask = mask.detach()
        if tracking:
            losses['depth'] = torch.abs(curr_data['depth'] - depth)[mask].sum()
        else:
            losses['depth'] = torch.abs(curr_data['depth'] - depth)[mask].mean()
    if flag_printloss:
        print("depth l1 loss is: ", losses['depth'])
    
    # RGB Loss
    if tracking and (use_sil_for_loss or ignore_outlier_depth_loss):
        color_mask = torch.tile(mask, (3, 1, 1))
        color_mask = color_mask.detach()
        losses['im'] = torch.abs(curr_data['im'] - im)[color_mask].sum()
    elif tracking:
        losses['im'] = torch.abs(curr_data['im'] - im).sum()
    else:
        losses['im'] = 0.8 * l1_loss_v1(im, curr_data['im']) + 0.2 * (1.0 - calc_ssim(im, curr_data['im']))
    if flag_printloss:
        print("rgb l1 loss is: ", losses['im'])

    # Visualize the Diff Images
    if tracking and visualize_tracking_loss:
        fig, ax = plt.subplots(2, 4, figsize=(12, 6))
        weighted_render_im = im * color_mask
        weighted_im = curr_data['im'] * color_mask
        weighted_render_depth = depth * mask
        weighted_depth = curr_data['depth'] * mask
        diff_rgb = torch.abs(weighted_render_im - weighted_im).mean(dim=0).detach().cpu()
        diff_depth = torch.abs(weighted_render_depth - weighted_depth).mean(dim=0).detach().cpu()
        viz_img = torch.clip(weighted_im.permute(1, 2, 0).detach().cpu(), 0, 1)
        ax[0, 0].imshow(viz_img)
        ax[0, 0].set_title("Weighted GT RGB")
        viz_render_img = torch.clip(weighted_render_im.permute(1, 2, 0).detach().cpu(), 0, 1)
        ax[1, 0].imshow(viz_render_img)
        ax[1, 0].set_title("Weighted Rendered RGB")
        ax[0, 1].imshow(weighted_depth[0].detach().cpu(), cmap="jet", vmin=0, vmax=6)
        ax[0, 1].set_title("Weighted GT Depth")
        ax[1, 1].imshow(weighted_render_depth[0].detach().cpu(), cmap="jet", vmin=0, vmax=6)
        ax[1, 1].set_title("Weighted Rendered Depth")
        ax[0, 2].imshow(diff_rgb, cmap="jet", vmin=0, vmax=0.8)
        ax[0, 2].set_title(f"Diff RGB, Loss: {torch.round(losses['im'])}")
        ax[1, 2].imshow(diff_depth, cmap="jet", vmin=0, vmax=0.8)
        ax[1, 2].set_title(f"Diff Depth, Loss: {torch.round(losses['depth'])}")
        ax[0, 3].imshow(presence_sil_mask.detach().cpu(), cmap="gray")
        ax[0, 3].set_title("Silhouette Mask")
        ax[1, 3].imshow(mask[0].detach().cpu(), cmap="gray")
        ax[1, 3].set_title("Loss Mask")
        # Turn off axis
        for i in range(2):
            for j in range(4):
                ax[i, j].axis('off')
        # Set Title
        fig.suptitle(f"Tracking Iteration: {tracking_iteration}", fontsize=16)
        # Figure Tight Layout
        fig.tight_layout()
        os.makedirs(plot_dir, exist_ok=True)
        plt.savefig(os.path.join(plot_dir, f"tmp.png"), bbox_inches='tight')
        plt.close()
        plot_img = cv2.imread(os.path.join(plot_dir, f"tmp.png"))
        cv2.imshow('Diff Images', plot_img)
        cv2.waitKey(1)
        ## Save Tracking Loss Viz
        # save_plot_dir = os.path.join(plot_dir, f"tracking_%04d" % iter_time_idx)
        # os.makedirs(save_plot_dir, exist_ok=True)
        # plt.savefig(os.path.join(save_plot_dir, f"%04d.png" % tracking_iteration), bbox_inches='tight')
        # plt.close()

    weighted_losses = {k: v * loss_weights[k] for k, v in losses.items()}
    loss = sum(weighted_losses.values())

    seen = radius > 0
    variables['max_2D_radius'][seen] = torch.max(radius[seen], variables['max_2D_radius'][seen])
    variables['seen'] = seen
    weighted_losses['loss'] = loss

    return loss, variables, weighted_losses

# with semantic (flat)
def get_loss_semantic(dataset, params, curr_data, variables, iter_time_idx, loss_weights, use_sil_for_loss,
             sil_thres, use_l1, ignore_outlier_depth_loss, vis_label_colorbar, tracking=False, 
             mapping=False, do_ba=False, plot_dir=None, visualize_semantic=False, visualize_tracking_loss=False, 
             tracking_iteration=None):
    
    flag_showtime = False
    flag_printloss = False
    
    # Initialize Loss Dictionary
    tracking_time_1 = time.time()
    losses = {}

    if tracking:
        # Get current frame Gaussians, where only the camera pose gets gradient
        transformed_gaussians = transform_to_frame(params, iter_time_idx, 
                                             gaussians_grad=False,
                                             camera_grad=True)
    elif mapping:
        if do_ba:
            # Get current frame Gaussians, where both camera pose and Gaussians get gradient
            transformed_gaussians = transform_to_frame(params, iter_time_idx,
                                                 gaussians_grad=True,
                                                 camera_grad=True)
        else:
            # Get current frame Gaussians, where only the Gaussians get gradient
            transformed_gaussians = transform_to_frame(params, iter_time_idx,
                                                 gaussians_grad=True,
                                                 camera_grad=False)
    else:
        # Get current frame Gaussians, where only the Gaussians get gradient
        transformed_gaussians = transform_to_frame(params, iter_time_idx,
                                             gaussians_grad=True,
                                             camera_grad=False)

    # Initialize Render Variables
    rendervar_sem = transformed_params2rendervar_semantic(params, transformed_gaussians)
    tracking_time_3 = time.time()

    # 1. Rendering
    rendervar_sem['means2D'].retain_grad()
    im, radius, im_semantic, rendered_depth, rendered_median_depth, rendered_final_opcity = Renderer_semantic(raster_settings=curr_data['cam'])(**rendervar_sem)


    variables['means2D'] = rendervar_sem['means2D']  
    tracking_time_4 = time.time()
    if flag_showtime:
        print("time usage -> render semantic {}*{}*{}: {} s".format( im_semantic.shape[0], im_semantic.shape[1], im_semantic.shape[2], (tracking_time_4 - tracking_time_3)) )

    # 2. Get Mask
    depth = rendered_depth
    presence_sil_mask = (rendered_final_opcity > sil_thres)
    
    # Mask with valid depth values (accounts for outlier depth values)
    nan_mask = (~torch.isnan(depth))
    if ignore_outlier_depth_loss:
        depth_error = torch.abs(curr_data['depth'] - depth) * (curr_data['depth'] > 0)
        mask = (depth_error < 10*depth_error.median())
        mask = mask & (curr_data['depth'] > 0)
    else:
        mask = (curr_data['depth'] > 0)
    mask = mask & nan_mask
    # Mask with presence silhouette mask (accounts for empty space)
    if tracking and use_sil_for_loss:
        mask = mask & presence_sil_mask

    # Depth loss
    if use_l1:
        mask = mask.detach()
        if tracking:
            losses['depth'] = torch.abs(curr_data['depth'] - depth)[mask].sum()
        else:
            losses['depth'] = torch.abs(curr_data['depth'] - depth)[mask].mean()
    if flag_printloss:
        print("depth l1 loss is: ", losses['depth'])
    
    # RGB Loss
    if tracking and (use_sil_for_loss or ignore_outlier_depth_loss):
        color_mask = torch.tile(mask, (3, 1, 1))
        color_mask = color_mask.detach()
        losses['im'] = torch.abs(curr_data['im'] - im)[color_mask].sum()
    elif tracking:
        losses['im'] = torch.abs(curr_data['im'] - im).sum()
    else:
        losses['im'] = 0.8 * l1_loss_v1(im, curr_data['im']) + 0.2 * (1.0 - calc_ssim(im, curr_data['im']))
    if flag_printloss:
        print("rgb l1 loss is: ", losses['im'])

    # Semantic Loss (only mapping)
    if not tracking:
        losses['sem'] = 0.0
        # print(dataset.num_semantic)   # 102
        if not isinstance(dataset.num_semantic, list):
            # original semantic class
            CrossEntropyLoss = torch.nn.CrossEntropyLoss()
            logits_2_label = lambda x: torch.argmax(torch.nn.functional.softmax(x, dim=-1),dim=-1)
            im_semantic = im_semantic.permute(1,2,0)                            # [num_classes, h, w] -> [h, w, num_classes]
            im_semantic_cal = im_semantic
            im_semantic_cal = im_semantic_cal.view(-1, im_semantic_cal.size(2)) 
            label_gt_semantic = curr_data['semantic_label_gt'].view(-1).long()
            losses['sem'] = CrossEntropyLoss(im_semantic_cal, label_gt_semantic)
        else:
            # tree class (replica) - mlp
            if "replica" in dataset.dataset_name:
                if 'tree' in dataset.sem_mode:

                    # ================ Cross-entropy Loss ====================== #
                    if True:
                        CrossEntropyLoss = torch.nn.CrossEntropyLoss()
                        logits_2_label = lambda x: torch.argmax(torch.nn.functional.softmax(x, dim=-1),dim=-1)

                        # Inter-level loss: sum of multi-level CE loss
                        if True:
                            for i_level in range(curr_data['semantic_label_gt'].shape[0]-1):
                                im_semantic_cal = transfer_tree_rendered_labelmap(im_semantic, i_level, dataset)
                                label_gt_semantic_level = curr_data['semantic_label_gt'][i_level, :, :].view(-1).long()
                                losses['sem'] += CrossEntropyLoss(im_semantic_cal, label_gt_semantic_level)     

            # tree class (scannet)          
            if "scannet" in dataset.dataset_name:
                # ================ Cross-entropy Loss ====================== #
                if True:
                    CrossEntropyLoss = torch.nn.CrossEntropyLoss()
                    logits_2_label = lambda x: torch.argmax(torch.nn.functional.softmax(x, dim=-1),dim=-1)

                    # Inter-level loss: sum of multi-level CE loss
                    for i_level in range(curr_data['semantic_label_gt'].shape[0]-1):
                        im_semantic_cal = transfer_tree_rendered_labelmap(im_semantic, i_level, dataset)
                        label_gt_semantic_level = curr_data['semantic_label_gt'][i_level, :, :].view(-1).long()
                        losses['sem'] += CrossEntropyLoss(im_semantic_cal, label_gt_semantic_level)
    
    weighted_losses = {k: v * loss_weights[k] for k, v in losses.items()}
    loss = sum(weighted_losses.values())

    seen = radius > 0
    variables['max_2D_radius'][seen] = torch.max(radius[seen], variables['max_2D_radius'][seen])
    variables['seen'] = seen
    weighted_losses['loss'] = loss

    return loss, variables, weighted_losses

# with semantic (tree)
def get_loss_semantic_mlp(dataset, params, curr_data, variables, iter_time_idx, loss_weights, use_sil_for_loss,
             sil_thres, use_l1, ignore_outlier_depth_loss, vis_label_colorbar, MLP_func, tracking=False, 
             mapping=False, do_ba=False, plot_dir=None, visualize_semantic=False, visualize_tracking_loss=False, 
             tracking_iteration=None):
    
    flag_showtime = False
    flag_printloss = False
    
    # Initialize Loss Dictionary
    tracking_time_1 = time.time()
    losses = {}

    if tracking:
        # Get current frame Gaussians, where only the camera pose gets gradient
        transformed_gaussians = transform_to_frame(params, iter_time_idx, 
                                             gaussians_grad=False,
                                             camera_grad=True)
    elif mapping:
        if do_ba:
            # Get current frame Gaussians, where both camera pose and Gaussians get gradient
            transformed_gaussians = transform_to_frame(params, iter_time_idx,
                                                 gaussians_grad=True,
                                                 camera_grad=True)
        else:
            # Get current frame Gaussians, where only the Gaussians get gradient
            transformed_gaussians = transform_to_frame(params, iter_time_idx,
                                                 gaussians_grad=True,
                                                 camera_grad=False)
    else:
        # Get current frame Gaussians, where only the Gaussians get gradient
        transformed_gaussians = transform_to_frame(params, iter_time_idx,
                                             gaussians_grad=True,
                                             camera_grad=False)

    # Initialize Render Variables
    rendervar_sem = transformed_params2rendervar_semantic(params, transformed_gaussians)
    tracking_time_3 = time.time()

    # 1. Rendering
    rendervar_sem['means2D'].retain_grad()
    im, radius, im_semantic, rendered_depth, rendered_median_depth, rendered_final_opcity = Renderer_semantic(raster_settings=curr_data['cam'])(**rendervar_sem)


    variables['means2D'] = rendervar_sem['means2D']  
    tracking_time_4 = time.time()
    if flag_showtime:
        print("time usage -> render semantic {}*{}*{}: {} fps".format( im_semantic.shape[0], im_semantic.shape[1], im_semantic.shape[2], 1.0/(tracking_time_4 - tracking_time_3)) )

    # 2. Get Mask
    depth = rendered_depth
    presence_sil_mask = (rendered_final_opcity > sil_thres)
    
    # Mask with valid depth values (accounts for outlier depth values)
    nan_mask = (~torch.isnan(depth))
    if ignore_outlier_depth_loss:
        depth_error = torch.abs(curr_data['depth'] - depth) * (curr_data['depth'] > 0)
        mask = (depth_error < 10*depth_error.median())
        mask = mask & (curr_data['depth'] > 0)
    else:
        mask = (curr_data['depth'] > 0)
    mask = mask & nan_mask
    # Mask with presence silhouette mask (accounts for empty space)
    if tracking and use_sil_for_loss:
        mask = mask & presence_sil_mask

    # Depth loss
    if use_l1:
        mask = mask.detach()
        if tracking:
            losses['depth'] = torch.abs(curr_data['depth'] - depth)[mask].sum()
        else:
            losses['depth'] = torch.abs(curr_data['depth'] - depth)[mask].mean()
    if flag_printloss:
        print("depth l1 loss is: ", losses['depth'])
    
    # RGB Loss
    if tracking and (use_sil_for_loss or ignore_outlier_depth_loss):
        color_mask = torch.tile(mask, (3, 1, 1))
        color_mask = color_mask.detach()
        losses['im'] = torch.abs(curr_data['im'] - im)[color_mask].sum()
    elif tracking:
        losses['im'] = torch.abs(curr_data['im'] - im).sum()
    else:
        losses['im'] = 0.8 * l1_loss_v1(im, curr_data['im']) + 0.2 * (1.0 - calc_ssim(im, curr_data['im']))
    if flag_printloss:
        print("rgb l1 loss is: ", losses['im'])

    # Semantic Loss (only mapping)
    if not tracking:
        losses['sem'] = 0.0
        if not isinstance(dataset.num_semantic, list):
            # original semantic class
            CrossEntropyLoss = torch.nn.CrossEntropyLoss()
            logits_2_label = lambda x: torch.argmax(torch.nn.functional.softmax(x, dim=-1),dim=-1)
            im_semantic = im_semantic.permute(1,2,0)           # [num_classes, h, w] -> [h, w, num_classes]
            im_semantic_cal = im_semantic
            im_semantic_cal = im_semantic_cal.view(-1, im_semantic_cal.size(2)) 
            label_gt_semantic = curr_data['semantic_label_gt'].view(-1).long()
            losses['sem'] = CrossEntropyLoss(im_semantic_cal, label_gt_semantic)
        else:
            # tree class (replica) - mlp
            if "replica" in dataset.dataset_name:
                if 'tree' in dataset.sem_mode:
                    weight_sem = [1.0, 5.0]

                    # ================ Cross-entropy Loss ====================== #
                    if True:
                        CrossEntropyLoss = torch.nn.CrossEntropyLoss()
                        logits_2_label = lambda x: torch.argmax(torch.nn.functional.softmax(x, dim=-1),dim=-1)

                        # Inter-level loss: sum of multi-level CE loss
                        if True:
                            loss_level = 0.0
                            for i_level in range(curr_data['semantic_label_gt'].shape[0]-1):
                                im_semantic_cal = transfer_tree_rendered_labelmap(im_semantic, i_level, dataset)
                                label_gt_semantic_level = curr_data['semantic_label_gt'][i_level, :, :].view(-1).long()
                                loss_level += CrossEntropyLoss(im_semantic_cal, label_gt_semantic_level)
                            losses['sem'] += weight_sem[0] * loss_level
                    
                    # ================ Cross-entropy Loss ====================== #
                    if curr_data['iter_mapping']>=14:
                        CrossEntropyLoss = torch.nn.CrossEntropyLoss()
                        im_semantic_in = im_semantic.unsqueeze(0)                           # [1, num_emdedding, h, w]]
                        logits = MLP_func(im_semantic_in)
                        logits = logits.squeeze(0).view(logits.shape[1], -1).permute(1, 0)             # [h*w, num_classes]  
                        label_gt_semantic = curr_data['semantic_label_gt'][-1, :, :].view(-1).long() 
                        loss_leaf = CrossEntropyLoss(logits, label_gt_semantic) 
                        losses['sem'] += weight_sem[1] * loss_leaf

            # tree class (scannet)          
            if "scannet" in dataset.dataset_name:
                if 'tree' in dataset.sem_mode:
                    weight_sem = [1.0, 5.0]

                    # ================ Cross-entropy Loss ====================== #
                    if True:
                        CrossEntropyLoss = torch.nn.CrossEntropyLoss()
                        logits_2_label = lambda x: torch.argmax(torch.nn.functional.softmax(x, dim=-1),dim=-1)

                        # tree class
                        # Inter-level loss: sum of multi-level CE loss
                        loss_level = 0.0
                        for i_level in range(curr_data['semantic_label_gt'].shape[0]-1):
                            im_semantic_cal = transfer_tree_rendered_labelmap(im_semantic, i_level, dataset)
                            label_gt_semantic_level = curr_data['semantic_label_gt'][i_level, :, :].view(-1).long()
                            loss_level += CrossEntropyLoss(im_semantic_cal, label_gt_semantic_level)
                            
                        losses['sem'] += weight_sem[0] * loss_level

                        if False:
                            print("losses['sem'] is: ", losses['sem'])
                    
                    # ================ Cross-level loss ====================== #
                    if curr_data['iter_mapping']>=14:
                        CrossEntropyLoss = torch.nn.CrossEntropyLoss()
                        im_semantic_in = im_semantic.unsqueeze(0)                       # [1, num_emdedding, h, w]]
                        logits = MLP_func(im_semantic_in)
                        logits = logits.squeeze(0).view(logits.shape[1], -1).permute(1, 0)             # [h*w, num_classes]  
                        label_gt_semantic = curr_data['semantic_label_gt'][-1, :, :].view(-1).long() 
                        loss_leaf = CrossEntropyLoss(logits, label_gt_semantic) 
                        losses['sem'] += weight_sem[1] * loss_leaf

        if flag_printloss:
            print("losses['sem'] is: ", losses['sem'])
    
    # Semantic visualization
    if visualize_semantic and curr_data['semantic_label_gt'].shape[2] == 1 and not tracking: 
        plot_dir = './experiments/ScanNet_semantic_ablation/test/'
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir, exist_ok=True)

        semantic_map_gt, num_classes_gt = label2map( curr_data['semantic_label_gt'].permute(2,0,1) ) # [h,w,1] -> [1,h,w]; semantic_map_gt: [num_classes, h, w]
        semantic_map_label = logits_2_label(im_semantic)     # [h, w]

        semantic_map_label = semantic_map_label.detach().cpu().numpy()
        semantic_map_label_colorbar = vis_label_colorbar[semantic_map_label]
        semantic_map_label_colorbar = semantic_map_label_colorbar.astype(np.uint8) 
        semantic_map_label_colorbar = np.squeeze(semantic_map_label_colorbar) 
        if 'iter_mapping' in curr_data:
            iter_current = curr_data['iter_mapping']
            cv2.imwrite(os.path.join(plot_dir, "sem_{}_map.png".format(iter_time_idx)), semantic_map_label_colorbar[...,::-1]) # bgr -> rgb 
        elif 'iter_tracking' in curr_data:
            iter_current = curr_data['iter_tracking']
            cv2.imwrite(os.path.join(plot_dir, "sem_{}_iter{}_track.png".format(iter_time_idx, curr_data['iter_tracking'])), semantic_map_label_colorbar[...,::-1]) # bgr -> rgb 
            
        # semantic gt
        if iter_current==0:
            gt_semantic_label = curr_data['semantic_label_gt'].detach().cpu().numpy().astype(int)
            label_gt_vis = vis_label_colorbar[gt_semantic_label]
            label_gt_vis = label_gt_vis.astype(np.uint8) 
            label_gt_vis = np.squeeze(label_gt_vis)
            cv2.imwrite(os.path.join(plot_dir, "gt_{}.png".format(iter_time_idx)),label_gt_vis[...,::-1]) # bgr -> rgb
        
    elif visualize_semantic and curr_data['semantic_label_gt'].shape[0] > 1 \
    and not curr_data['semantic_label_gt'].shape[0]==curr_data['cam'][0] and not tracking:
        if 'tree' in dataset.sem_mode:  
            # tree visualization
            if "replica" in dataset.dataset_name:
                plot_dir = './experiments/Replica_semantic_mlp/test/'
            elif "scannet" in dataset.dataset_name:
                plot_dir = './experiments/ScanNet_semantic_ablation/test/'

            if not os.path.exists(plot_dir):
                os.makedirs(plot_dir, exist_ok=True)

            # 1. estimation save  ====================
            im_semantic_in = im_semantic.unsqueeze(0)                           # [1, num_emdedding, h, w]]
            logits = MLP_func(im_semantic_in).squeeze(0)                        # [num_classes, h, w]
            logits = F.softmax(logits, dim=0)
            im_semantic_label = torch.argmax(logits, dim=0)                     # estimated label map

            im_semantic_label_vis = im_semantic_label.detach().cpu().numpy()
            im_semantic_label_colorbar = vis_label_colorbar[im_semantic_label_vis].astype(np.uint8) 
            im_semantic_label_colorbar = np.squeeze(im_semantic_label_colorbar) 
            save_name = "sem_{:04d}".format(iter_time_idx)
            print("===> save at: ", os.path.join(plot_dir,  save_name+".png"))
            cv2.imwrite(os.path.join(plot_dir,  save_name+".png"), im_semantic_label_colorbar[...,::-1])

            # 2. gt_semantic_label save  ====================
            if "scannet" in dataset.dataset_name:
                label_gt_original = curr_data['semantic_label_gt'][-1,:,:]
                save_name = "gt_{:04d}".format(iter_time_idx)
                label_gt_original = label_gt_original.detach().cpu().numpy().astype(int)
                im_semantic_treelabel_gt_vis = vis_label_colorbar[label_gt_original].astype(np.uint8) 
                im_semantic_treelabel_gt_vis = np.squeeze(im_semantic_treelabel_gt_vis)
            elif "replica" in dataset.dataset_name:
                label_gt_use = curr_data['semantic_label_gt'][:-1,:,:]
                save_name = "gt_{:04d}".format(iter_time_idx)
                im_semantic_treelabel_gt = label_gt_use.detach().cpu()
                im_semantic_treelabel_gt_vis = semantic_label_vis_replica(im_semantic_treelabel_gt, dataset.colour_map_np_level, iter = curr_data["iter_mapping"])

            print("===> save at: ", os.path.join(plot_dir,  save_name+".png"))
            cv2.imwrite(os.path.join(plot_dir,  save_name+".png"), im_semantic_treelabel_gt_vis[...,::-1])

            # eval (replica-tree)
            if "replica" in dataset.dataset_name:
                thresh_iter_end = 59
            elif "scannet" in dataset.dataset_name:
                thresh_iter_end = 29
            if curr_data['iter_mapping']>=thresh_iter_end and True:
                print("current-id is: ", curr_data['id'])
                eval_semantic_single(dataset, im_semantic_label, curr_data['semantic_label_gt'][-1, :, :])

    weighted_losses = {k: v * loss_weights[k] for k, v in losses.items()}
    loss = sum(weighted_losses.values())

    seen = radius > 0
    variables['max_2D_radius'][seen] = torch.max(radius[seen], variables['max_2D_radius'][seen])
    variables['seen'] = seen
    weighted_losses['loss'] = loss

    return loss, variables, weighted_losses


def initialize_new_params(new_pt_cld, mean3_sq_dist, gaussian_distribution):
    num_pts = new_pt_cld.shape[0]
    means3D = new_pt_cld[:, :3] # [num_gaussians, 3]
    unnorm_rots = np.tile([1, 0, 0, 0], (num_pts, 1)) # [num_gaussians, 4]
    logit_opacities = torch.zeros((num_pts, 1), dtype=torch.float, device="cuda")
    if gaussian_distribution == "isotropic":
        log_scales = torch.tile(torch.log(torch.sqrt(mean3_sq_dist))[..., None], (1, 1))
    elif gaussian_distribution == "anisotropic":
        log_scales = torch.tile(torch.log(torch.sqrt(mean3_sq_dist))[..., None], (1, 3))
    else:
        raise ValueError(f"Unknown gaussian_distribution {gaussian_distribution}")
    params = {
        'means3D': means3D,
        'rgb_colors': new_pt_cld[:, 3:6],
        'unnorm_rotations': unnorm_rots,
        'logit_opacities': logit_opacities,
        'log_scales': log_scales,
    }
    for k, v in params.items():
        # Check if value is already a torch tensor
        if not isinstance(v, torch.Tensor):
            params[k] = torch.nn.Parameter(torch.tensor(v).cuda().float().contiguous().requires_grad_(True))
        else:
            params[k] = torch.nn.Parameter(v.cuda().float().contiguous().requires_grad_(True))

    return params

def initialize_new_params_semantic(new_pt_cld, mean3_sq_dist, num_labels):
    # 0: initial all 0; 1: initial randn; 2: others?
    flag_init = 2

    num_pts = new_pt_cld.shape[0]
    means3D = new_pt_cld[:, :3] # [num_gaussians, 3]
    unnorm_rots = np.tile([1, 0, 0, 0], (num_pts, 1)) # [num_gaussians, 3]
    logit_opacities = torch.zeros((num_pts, 1), dtype=torch.float, device="cuda")
    if flag_init == 0:
        logit_semantic = torch.zeros((num_pts, num_labels), dtype=torch.float, device="cuda")   # [num_gaussians, num_labels]
    elif flag_init == 1:
        logit_semantic = RGB2SH(torch.rand((num_pts, num_labels), device="cuda"))
    elif flag_init == 2:
        logit_semantic = torch.rand((num_pts, num_labels), device="cuda")
    
    params = {
        'means3D': means3D,
        'rgb_colors': new_pt_cld[:, 3:6],
        'unnorm_rotations': unnorm_rots,
        'logit_opacities': logit_opacities,
        'log_scales': torch.tile(torch.log(torch.sqrt(mean3_sq_dist))[..., None], (1, 1)),
        'semantic': logit_semantic,
    }
    for k, v in params.items():
        # Check if value is already a torch tensor
        if not isinstance(v, torch.Tensor):
            params[k] = torch.nn.Parameter(torch.tensor(v).cuda().float().contiguous().requires_grad_(True))
        else:
            params[k] = torch.nn.Parameter(v.cuda().float().contiguous().requires_grad_(True))

    return params

def add_new_gaussians(params, variables, curr_data, sil_thres, 
                      time_idx, mean_sq_dist_method, gaussian_distribution):
    # Silhouette Rendering
    transformed_gaussians = transform_to_frame(params, time_idx, gaussians_grad=False, camera_grad=False)
    depth_sil_rendervar = transformed_params2depthplussilhouette(params, curr_data['w2c'],
                                                                 transformed_gaussians)
    depth_sil, _, _, = Renderer(raster_settings=curr_data['cam'])(**depth_sil_rendervar)
    silhouette = depth_sil[1, :, :]
    non_presence_sil_mask = (silhouette < sil_thres)
    # Check for new foreground objects by using GT depth
    gt_depth = curr_data['depth'][0, :, :]
    render_depth = depth_sil[0, :, :]
    depth_error = torch.abs(gt_depth - render_depth) * (gt_depth > 0)
    non_presence_depth_mask = (render_depth > gt_depth) * (depth_error > 50*depth_error.median())
    # Determine non-presence mask
    non_presence_mask = non_presence_sil_mask | non_presence_depth_mask
    # Flatten mask
    non_presence_mask = non_presence_mask.reshape(-1)

    # Get the new frame Gaussians based on the Silhouette
    if torch.sum(non_presence_mask) > 0:
        # Get the new pointcloud in the world frame
        curr_cam_rot = torch.nn.functional.normalize(params['cam_unnorm_rots'][..., time_idx].detach())
        curr_cam_tran = params['cam_trans'][..., time_idx].detach()
        curr_w2c = torch.eye(4).cuda().float()
        curr_w2c[:3, :3] = build_rotation(curr_cam_rot)
        curr_w2c[:3, 3] = curr_cam_tran
        valid_depth_mask = (curr_data['depth'][0, :, :] > 0)
        non_presence_mask = non_presence_mask & valid_depth_mask.reshape(-1)
        new_pt_cld, mean3_sq_dist = get_pointcloud(curr_data['im'], curr_data['depth'], curr_data['intrinsics'], 
                                    curr_w2c, mask=non_presence_mask, compute_mean_sq_dist=True,
                                    mean_sq_dist_method=mean_sq_dist_method)
        new_params = initialize_new_params(new_pt_cld, mean3_sq_dist, gaussian_distribution)
        for k, v in new_params.items():
            params[k] = torch.nn.Parameter(torch.cat((params[k], v), dim=0).requires_grad_(True))
        num_pts = params['means3D'].shape[0]
        variables['means2D_gradient_accum'] = torch.zeros(num_pts, device="cuda").float()
        variables['denom'] = torch.zeros(num_pts, device="cuda").float()
        variables['max_2D_radius'] = torch.zeros(num_pts, device="cuda").float()
        new_timestep = time_idx*torch.ones(new_pt_cld.shape[0],device="cuda").float()
        variables['timestep'] = torch.cat((variables['timestep'],new_timestep),dim=0)

    return params, variables


def add_new_gaussians_newtest(params, variables, curr_data, sil_thres, 
                      time_idx, mean_sq_dist_method, gaussian_distribution, flag_use_render=1):
    # Silhouette Rendering
    transformed_gaussians = transform_to_frame(params, time_idx, gaussians_grad=False, camera_grad=False)
    rendervar = transformed_params2rendervar(params, transformed_gaussians)
    if flag_use_render==1:
        im, radius, rendered_depth, rendered_median_depth, rendered_final_opcity, rendered_mask = Renderer(raster_settings=curr_data['cam'])(**rendervar)
    elif flag_use_render==2:
        im, radius, rendered_depth, rendered_final_opcity = Renderer(raster_settings=curr_data['cam'])(**rendervar)
    
    # silhouette = rendered_mask
    silhouette = rendered_final_opcity
    non_presence_sil_mask = (silhouette < sil_thres)

    # Check for new foreground objects by using GT depth
    gt_depth = curr_data['depth'][0, :, :]
    render_depth = rendered_depth
    depth_error = torch.abs(gt_depth - render_depth) * (gt_depth > 0)
    non_presence_depth_mask = (render_depth > gt_depth) * (depth_error > 50*depth_error.median())
    # Determine non-presence mask
    non_presence_mask = non_presence_sil_mask | non_presence_depth_mask
    # Flatten mask
    non_presence_mask = non_presence_mask.reshape(-1)

    # Get the new frame Gaussians based on the Silhouette
    if torch.sum(non_presence_mask) > 0:
        # Get the new pointcloud in the world frame
        curr_cam_rot = torch.nn.functional.normalize(params['cam_unnorm_rots'][..., time_idx].detach())
        curr_cam_tran = params['cam_trans'][..., time_idx].detach()
        curr_w2c = torch.eye(4).cuda().float()
        curr_w2c[:3, :3] = build_rotation(curr_cam_rot)
        curr_w2c[:3, 3] = curr_cam_tran
        valid_depth_mask = (curr_data['depth'][0, :, :] > 0)
        non_presence_mask = non_presence_mask & valid_depth_mask.reshape(-1)
        new_pt_cld, mean3_sq_dist = get_pointcloud(curr_data['im'], curr_data['depth'], curr_data['intrinsics'], 
                                    curr_w2c, mask=non_presence_mask, compute_mean_sq_dist=True,
                                    mean_sq_dist_method=mean_sq_dist_method)
        new_params = initialize_new_params(new_pt_cld, mean3_sq_dist, gaussian_distribution)
        for k, v in new_params.items():
            params[k] = torch.nn.Parameter(torch.cat((params[k], v), dim=0).requires_grad_(True))
        num_pts = params['means3D'].shape[0]
        variables['means2D_gradient_accum'] = torch.zeros(num_pts, device="cuda").float()
        variables['denom'] = torch.zeros(num_pts, device="cuda").float()
        variables['max_2D_radius'] = torch.zeros(num_pts, device="cuda").float()
        new_timestep = time_idx*torch.ones(new_pt_cld.shape[0],device="cuda").float()
        variables['timestep'] = torch.cat((variables['timestep'],new_timestep),dim=0)

    return params, variables


def add_new_gaussians_semantic(params, variables, curr_data, sil_thres, time_idx, mean_sq_dist_method, num_semantic):
    # Silhouette Rendering
    transformed_pts = transform_to_frame(params, time_idx, gaussians_grad=False, camera_grad=False)
    depth_sil_rendervar = transformed_params2depthplussilhouette(params, curr_data['w2c'],
                                                                 transformed_pts)
    depth_sil, _, _, = Renderer(raster_settings=curr_data['cam'])(**depth_sil_rendervar)
    silhouette = depth_sil[1, :, :]                         # depth_sil: [3,h,w] -> 3-dim:[depth, silhouette, ?]
    non_presence_sil_mask = (silhouette < sil_thres)
    # Check for new foreground objects by using GT depth
    gt_depth = curr_data['depth'][0, :, :]
    render_depth = depth_sil[0, :, :]
    depth_error = torch.abs(gt_depth - render_depth) * (gt_depth > 0)
    non_presence_depth_mask = (render_depth > gt_depth) * (depth_error > 50*depth_error.median())
    # Determine non-presence mask
    non_presence_mask = non_presence_sil_mask | non_presence_depth_mask
    # Flatten mask
    non_presence_mask = non_presence_mask.reshape(-1)

    # Get the new frame Gaussians based on the Silhouette
    if torch.sum(non_presence_mask) > 0:
        # Get the new pointcloud in the world frame
        curr_cam_rot = torch.nn.functional.normalize(params['cam_unnorm_rots'][..., time_idx].detach())
        curr_cam_tran = params['cam_trans'][..., time_idx].detach()
        curr_w2c = torch.eye(4).cuda().float()
        curr_w2c[:3, :3] = build_rotation(curr_cam_rot)
        curr_w2c[:3, 3] = curr_cam_tran
        valid_depth_mask = (curr_data['depth'][0, :, :] > 0)
        non_presence_mask = non_presence_mask & valid_depth_mask.reshape(-1)
        new_pt_cld, mean3_sq_dist = get_pointcloud(curr_data['im'], curr_data['depth'], curr_data['intrinsics'], 
                                    curr_w2c, mask=non_presence_mask, compute_mean_sq_dist=True,
                                    mean_sq_dist_method=mean_sq_dist_method)
        new_params = initialize_new_params_semantic(new_pt_cld, mean3_sq_dist, num_semantic)        # [sem] change
        for k, v in new_params.items():
            params[k] = torch.nn.Parameter(torch.cat((params[k], v), dim=0).requires_grad_(True))
        num_pts = params['means3D'].shape[0]
        variables['means2D_gradient_accum'] = torch.zeros(num_pts, device="cuda").float()
        variables['denom'] = torch.zeros(num_pts, device="cuda").float()
        variables['max_2D_radius'] = torch.zeros(num_pts, device="cuda").float()
        new_timestep = time_idx*torch.ones(new_pt_cld.shape[0],device="cuda").float()
        variables['timestep'] = torch.cat((variables['timestep'],new_timestep),dim=0)

    return params, variables

def add_new_gaussians_semantic_newrender(params, variables, curr_data, sil_thres, time_idx, 
                                         mean_sq_dist_method, num_semantic, flag_use_render=1):
    # Silhouette Rendering
    transformed_pts = transform_to_frame(params, time_idx, gaussians_grad=False, camera_grad=False)
    rendervar_sem = transformed_params2rendervar_semantic(params, transformed_pts)

    if flag_use_render==1:
        im, radius, im_semantic, rendered_depth, rendered_median_depth, rendered_final_opcity =\
        Renderer_semantic(raster_settings=curr_data['cam'])(**rendervar_sem)

    silhouette = rendered_final_opcity.squeeze(0)
    non_presence_sil_mask = (silhouette < sil_thres)
    # Check for new foreground objects by using GT depth
    gt_depth = curr_data['depth'][0, :, :]
    render_depth = rendered_depth.squeeze(0)
    depth_error = torch.abs(gt_depth - render_depth) * (gt_depth > 0)
    non_presence_depth_mask = (render_depth > gt_depth) * (depth_error > 50*depth_error.median())
    # Determine non-presence mask
    non_presence_mask = non_presence_sil_mask | non_presence_depth_mask
    # Flatten mask
    non_presence_mask = non_presence_mask.reshape(-1)

    # Get the new frame Gaussians based on the Silhouette
    if torch.sum(non_presence_mask) > 0:
        # Get the new pointcloud in the world frame
        curr_cam_rot = torch.nn.functional.normalize(params['cam_unnorm_rots'][..., time_idx].detach())
        curr_cam_tran = params['cam_trans'][..., time_idx].detach()
        curr_w2c = torch.eye(4).cuda().float()
        curr_w2c[:3, :3] = build_rotation(curr_cam_rot)
        curr_w2c[:3, 3] = curr_cam_tran
        valid_depth_mask = (curr_data['depth'][0, :, :] > 0)
        non_presence_mask = non_presence_mask & valid_depth_mask.reshape(-1)
        new_pt_cld, mean3_sq_dist = get_pointcloud(curr_data['im'], curr_data['depth'], curr_data['intrinsics'], 
                                    curr_w2c, mask=non_presence_mask, compute_mean_sq_dist=True,
                                    mean_sq_dist_method=mean_sq_dist_method)
        new_params = initialize_new_params_semantic(new_pt_cld, mean3_sq_dist, num_semantic)        # [sem] change
        for k, v in new_params.items():
            params[k] = torch.nn.Parameter(torch.cat((params[k], v), dim=0).requires_grad_(True))
        num_pts = params['means3D'].shape[0]
        variables['means2D_gradient_accum'] = torch.zeros(num_pts, device="cuda").float()
        variables['denom'] = torch.zeros(num_pts, device="cuda").float()
        variables['max_2D_radius'] = torch.zeros(num_pts, device="cuda").float()
        new_timestep = time_idx*torch.ones(new_pt_cld.shape[0],device="cuda").float()
        variables['timestep'] = torch.cat((variables['timestep'],new_timestep),dim=0)

    return params, variables

def initialize_camera_pose(params, curr_time_idx, forward_prop):
    with torch.no_grad():
        if curr_time_idx > 1 and forward_prop:
            # Initialize the camera pose for the current frame based on a constant velocity model
            # Rotation
            prev_rot1 = F.normalize(params['cam_unnorm_rots'][..., curr_time_idx-1].detach())
            prev_rot2 = F.normalize(params['cam_unnorm_rots'][..., curr_time_idx-2].detach())
            new_rot = F.normalize(prev_rot1 + (prev_rot1 - prev_rot2))
            params['cam_unnorm_rots'][..., curr_time_idx] = new_rot.detach()
            # Translation
            prev_tran1 = params['cam_trans'][..., curr_time_idx-1].detach()
            prev_tran2 = params['cam_trans'][..., curr_time_idx-2].detach()
            new_tran = prev_tran1 + (prev_tran1 - prev_tran2)
            params['cam_trans'][..., curr_time_idx] = new_tran.detach()
        else:
            # Initialize the camera pose for the current frame
            params['cam_unnorm_rots'][..., curr_time_idx] = params['cam_unnorm_rots'][..., curr_time_idx-1].detach()
            params['cam_trans'][..., curr_time_idx] = params['cam_trans'][..., curr_time_idx-1].detach()
    
    return params

def convert_params_to_store(params):
    params_to_store = {}
    for k, v in params.items():
        if isinstance(v, torch.Tensor):
            params_to_store[k] = v.detach().clone()
        else:
            params_to_store[k] = v
    return params_to_store

def label2map(label_gt, num_semantic=-1):
    # label_gt [1,h,w]
    # label_map [num_classes,h,w]
    
    height, width = label_gt.shape[1], label_gt.shape[2]
    if num_semantic<0:
        max_class = torch.max(label_gt).item()
        num_classes = max_class+1
        num_classes = int(num_classes)
    else:
        num_classes = num_semantic

    label_map_list = []
    for i_sem in range(0,num_classes):
        label_map_single = torch.zeros((1, height, width), dtype=torch.float, device="cuda")
        label_map_single[label_gt==i_sem] = 1.0
        label_map_single = label_map_single.view(1,height, width)
        label_map_list.append(label_map_single)

    label_map = torch.cat(label_map_list,dim=0)

    return label_map, num_classes

def label2map_tree(label_gt, num_classes_tree):
    """
    label_gt [4, 480, 640]
    num_classes_tree: [level0_num_classes, level1_num_classes, ...]
    num_classes = sum(num_classes_tree[:-1])
    label_map: [num_classes, h, w]
    """

    num_classes_tree_use = num_classes_tree[:-1]
    height, width = label_gt.shape[1], label_gt.shape[2]
    num_classes = sum(num_classes_tree_use)

    label_map_list = []
    for i_level in range(len(num_classes_tree_use)):
        num_classes_thislevel = num_classes_tree_use[i_level]
        # print( "level-{}, num_classes_thislevel: {}".format(i_level, num_classes_thislevel) )
        label_gt_level = label_gt[i_level, :, :].unsqueeze(0)
        # print( label_gt_level.max() , label_gt[i_level, :, :].max())
        for i_sem in range(0, num_classes_thislevel):
            # print("     for i_sem: ", i_sem)
            label_map_single = torch.zeros((1, height, width), dtype=torch.float, device="cuda")
            label_map_single[label_gt_level==i_sem] = 1.0
            label_map_single = label_map_single.view(1,height, width)
            label_map_list.append(label_map_single)
    label_map = torch.cat(label_map_list,dim=0)
    assert(label_map.shape[0]==num_classes)

    return label_map, num_classes

def label2map_tree_new(label_gt, num_classes_tree):
    """
    Transfer to one-hot : 
    label_gt [num_level, h, w]
    num_classes_tree: [level0_num_classes, level1_num_classes, ...]
    num_classes = sum(num_classes_tree[:-1])
    RETURN: label_map: [num_classes, h, w] 
    """
    flag_debug = False

    num_classes_tree_use = num_classes_tree[:-1]
    height, width = label_gt.shape[1], label_gt.shape[2]
    num_classes = sum(num_classes_tree_use)

    label_map_list = []
    # tree-levels
    for i_level in range(len(num_classes_tree_use)):        # each level: generate one-hot map
        
        num_classes_thislevel = num_classes_tree_use[i_level]
        if flag_debug:
            print( "level-{}, num_classes_thislevel: {}".format(i_level, num_classes_thislevel) )
        label_gt_level = label_gt[i_level, :, :].unsqueeze(0)
        if flag_debug:
            print("original: ")
            print( label_gt_level.min())
            print( label_gt_level.max())
            
        for i_sem in range(0, num_classes_thislevel):   # each semantic label in each level: scan (4: [0,1,2,3])
            if flag_debug:
                print("     for i_sem: ", i_sem)
            label_map_single = torch.zeros((1, height, width), dtype=torch.float, device="cuda")
            label_map_single[label_gt_level==i_sem] = 1.0
            label_map_single = label_map_single.view(1,height, width)
            label_map_list.append(label_map_single)
        
        if flag_debug:
            print("     current shape label_map_list: ", len(label_map_list))

    label_map = torch.cat(label_map_list,dim=0)
    assert(label_map.shape[0]==num_classes)

    if flag_debug:
        idx_test = [torch.tensor([88,82]), torch.tensor([38,26]), torch.tensor([168,32]), torch.tensor([350,584])]
        print("label_map shape is: ", label_map.shape)
        for item_idx in range(len(idx_test)):
            pixel_idx = idx_test[item_idx]
            print("pixel_idx is: ", pixel_idx)
            print(label_gt[:, pixel_idx[0], pixel_idx[1]])
            print(label_map[:, pixel_idx[0], pixel_idx[1]])

    return label_map, num_classes


def addnoise(input_data):
    noise_add = torch.rand(input_data.shape, device="cuda")
    output_data = input_data + noise_add

    return output_data

def hierslam_main(config: dict):

    # Print Config
    print("Loaded Config:")
    if "use_depth_loss_thres" not in config['tracking']:
        config['tracking']['use_depth_loss_thres'] = False
        config['tracking']['depth_loss_thres'] = 100000
    if "visualize_tracking_loss" not in config['tracking']:
        config['tracking']['visualize_tracking_loss'] = False
    if "gaussian_distribution" not in config:
        config['gaussian_distribution'] = "isotropic"
    print(f"{config}")

    # Create Output Directories
    output_dir = os.path.join(config["workdir"], config["run_name"])
    eval_dir = os.path.join(output_dir, "eval")
    os.makedirs(eval_dir, exist_ok=True)
    
    # Init WandB
    if config['use_wandb']:
        wandb_time_step = 0
        wandb_tracking_step = 0
        wandb_mapping_step = 0
        wandb_run = wandb.init(project=config['wandb']['project'],
                               entity=config['wandb']['entity'],
                               group=config['wandb']['group'],
                               name=config['wandb']['name'],
                               config=config)
        with open(os.path.join(output_dir,'wandb_path.txt'),'w') as f: 
            f.writelines(wandb_run.url) 
            f.writelines("\n")
            f.writelines(wandb_run.dir) 

    # Get Device
    device = torch.device(config["primary_device"])

    # Load Dataset
    print("Loading Dataset ...")
    dataset_config = config["data"]
    if "gradslam_data_cfg" not in dataset_config:
        gradslam_data_cfg = {}
        gradslam_data_cfg["dataset_name"] = dataset_config["dataset_name"]
    else:
        gradslam_data_cfg = load_dataset_config(dataset_config["gradslam_data_cfg"])
        gradslam_data_cfg = {**gradslam_data_cfg, **dataset_config}
    if "ignore_bad" not in dataset_config:
        dataset_config["ignore_bad"] = False
    if "use_train_split" not in dataset_config:
        dataset_config["use_train_split"] = True
    if "densification_image_height" not in dataset_config:
        dataset_config["densification_image_height"] = dataset_config["desired_image_height"]
        dataset_config["densification_image_width"] = dataset_config["desired_image_width"]
        seperate_densification_res = False
    else:
        if dataset_config["densification_image_height"] != dataset_config["desired_image_height"] or \
            dataset_config["densification_image_width"] != dataset_config["desired_image_width"]:
            seperate_densification_res = True
        else:
            seperate_densification_res = False
    if "tracking_image_height" not in dataset_config:
        dataset_config["tracking_image_height"] = dataset_config["desired_image_height"]
        dataset_config["tracking_image_width"] = dataset_config["desired_image_width"]
        seperate_tracking_res = False
    else:
        if dataset_config["tracking_image_height"] != dataset_config["desired_image_height"] or \
            dataset_config["tracking_image_width"] != dataset_config["desired_image_width"]:
            seperate_tracking_res = True
        else:
            seperate_tracking_res = False

    # semantic
    gradslam_data_cfg["results_dir"] = config["results_dir"]
    flag_use_semantic = False
    if "semantic" in gradslam_data_cfg["dataset_name"]:
        print(" ********** USE SEMANTIC ********** ")
        flag_use_semantic = True
        gradslam_data_cfg["sem_mode"] = config["data"]["sem_mode"]
        gradslam_data_cfg["model"] = config["model"]

    # Poses are relative to the first frame
    dataset = get_dataset(
        config_dict=gradslam_data_cfg,
        basedir=dataset_config["basedir"],
        sequence=os.path.basename(dataset_config["sequence"]),
        start=dataset_config["start"],
        end=dataset_config["end"],
        stride=dataset_config["stride"],
        desired_height=dataset_config["desired_image_height"],
        desired_width=dataset_config["desired_image_width"],
        device=device,
        relative_pose=True,
        ignore_bad=dataset_config["ignore_bad"],
        use_train_split=dataset_config["use_train_split"],
    )
    num_frames = dataset_config["num_frames"]
    if num_frames == -1:
        num_frames = len(dataset)

    if flag_use_semantic:
        if "semantic" in gradslam_data_cfg["dataset_name"]:
            if "scannet" in gradslam_data_cfg["dataset_name"]:
                # scannet
                if dataset.sem_mode == "binary1" or dataset.sem_mode == "eigen13" or dataset.sem_mode == "nyu40":
                    num_semantic = int(re.findall(r"\d+", config["data"]["sem_mode"])[0])+1
                elif "tree" in dataset.sem_mode:
                    num_semantic = sum(dataset.num_semantic[:-1]) # +1: void
                elif dataset.sem_mode == "raw":
                    num_semantic = dataset.num_semantic
            elif "replica" in gradslam_data_cfg["dataset_name"]:
                # replica
                if dataset.sem_mode == "original":
                    num_semantic = dataset.num_semantic
                if "tree" in dataset.sem_mode:
                    num_semantic = sum(dataset.num_semantic[:-1])
                if "model" in dataset.sem_mode:
                    num_semantic = dataset.num_semantic
        print("num_semantic is: ", num_semantic)        
        
    # for visualization  
    if flag_use_semantic and "scannet" in gradslam_data_cfg["dataset_name"]:
        vis_label_colorbar = dataset.colour_map_np
    elif flag_use_semantic:
        vis_label_colorbar = dataset.colors_map_all
    print('****** sequence frame number is: ', num_frames)

    # Init seperate dataloader for densification if required
    if not flag_use_semantic:
        if seperate_densification_res:
            densify_dataset = get_dataset(
                config_dict=gradslam_data_cfg,
                basedir=dataset_config["basedir"],
                sequence=os.path.basename(dataset_config["sequence"]),
                start=dataset_config["start"],
                end=dataset_config["end"],
                stride=dataset_config["stride"],
                desired_height=dataset_config["densification_image_height"],
                desired_width=dataset_config["densification_image_width"],
                device=device,
                relative_pose=True,
                ignore_bad=dataset_config["ignore_bad"],
                use_train_split=dataset_config["use_train_split"],
            )
            # Initialize Parameters, Canonical & Densification Camera parameters
            params, variables, intrinsics, first_frame_w2c, cam, \
                densify_intrinsics, densify_cam = initialize_first_timestep(dataset, num_frames,
                                                                            config['scene_radius_depth_ratio'],
                                                                            config['mean_sq_dist_method'],
                                                                            densify_dataset=densify_dataset,
                                                                            gaussian_distribution=config['gaussian_distribution'])                                                                                                                  
        else:
            # Initialize Parameters & Canoncial Camera parameters
            params, variables, intrinsics, first_frame_w2c, cam = initialize_first_timestep(dataset, num_frames, 
                                                                                            config['scene_radius_depth_ratio'],
                                                                                            config['mean_sq_dist_method'],
                                                                                            gaussian_distribution=config['gaussian_distribution'])
    else:
        if "scannet" in gradslam_data_cfg["dataset_name"]:
            # scannet
            # Initialize Parameters & Canoncial Camera parameters (flat representation/tree representation)
            if dataset.sem_mode == "binary1" or dataset.sem_mode == "eigen13" or dataset.sem_mode == "nyu40" or dataset.sem_mode == "raw":
                params, variables, intrinsics, first_frame_w2c, cam = initialize_first_timestep_semantic(dataset, num_frames, num_semantic,
                                                                                            config['scene_radius_depth_ratio'],
                                                                                            config['mean_sq_dist_method'])
            else:
                if dataset.use_pyramid:
                    params, variables, intrinsics, intrinsics_py, first_frame_w2c, cam, cam_py = initialize_first_timestep_semantic_tree(dataset, num_frames, 
                                                                                            config['scene_radius_depth_ratio'],
                                                                                            config['mean_sq_dist_method'])
                else:
                    params, variables, intrinsics, first_frame_w2c, cam = initialize_first_timestep_semantic_tree(dataset, num_frames, 
                                                                                            config['scene_radius_depth_ratio'],
                                                                                            config['mean_sq_dist_method'])
        elif "replica" in gradslam_data_cfg["dataset_name"]:
            # replica
            if dataset.sem_mode == "original":
                params, variables, intrinsics, first_frame_w2c, cam = initialize_first_timestep_semantic(dataset, num_frames, num_semantic,
                                                                                            config['scene_radius_depth_ratio'],
                                                                                            config['mean_sq_dist_method'])
            elif "tree" in dataset.sem_mode:
                params, variables, intrinsics, first_frame_w2c, cam = initialize_first_timestep_semantic_tree(dataset, num_frames, 
                                                                                            config['scene_radius_depth_ratio'],
                                                                                            config['mean_sq_dist_method'])

    
    # Init seperate dataloader for tracking if required
    if seperate_tracking_res:
        tracking_dataset = get_dataset(
            config_dict=gradslam_data_cfg,
            basedir=dataset_config["basedir"],
            sequence=os.path.basename(dataset_config["sequence"]),
            start=dataset_config["start"],
            end=dataset_config["end"],
            stride=dataset_config["stride"],
            desired_height=dataset_config["tracking_image_height"],
            desired_width=dataset_config["tracking_image_width"],
            device=device,
            relative_pose=True,
            ignore_bad=dataset_config["ignore_bad"],
            use_train_split=dataset_config["use_train_split"],
        )
        tracking_color, _, tracking_intrinsics, _ = tracking_dataset[0]
        tracking_color = tracking_color.permute(2, 0, 1) / 255 # (H, W, C) -> (C, H, W)
        tracking_intrinsics = tracking_intrinsics[:3, :3]
        tracking_cam = setup_camera(tracking_color.shape[2], tracking_color.shape[1], 
                            tracking_intrinsics.cpu().numpy(), first_frame_w2c.detach().cpu().numpy())
    
    # Initialize list to keep track of Keyframes
    keyframe_list = []
    keyframe_time_indices = []
    
    # Init Variables to keep track of ground truth poses and runtimes
    gt_w2c_all_frames = []
    tracking_iter_time_sum = 0
    tracking_iter_time_count = 0
    mapping_iter_time_sum = 0
    mapping_iter_time_count = 0
    tracking_frame_time_sum = 0
    tracking_frame_time_count = 0
    mapping_frame_time_sum = 0
    mapping_frame_time_count = 0

    # Load Checkpoint
    if config['load_checkpoint']: 
        checkpoint_time_idx = config['checkpoint_time_idx']
        print(f"Loading Checkpoint for Frame {checkpoint_time_idx}")
        ckpt_path = os.path.join(config['workdir'], config['run_name'], f"params{checkpoint_time_idx}.npz")
        params = dict(np.load(ckpt_path, allow_pickle=True))
        params = {k: torch.tensor(params[k]).cuda().float().requires_grad_(True) for k in params.keys()}
        variables['max_2D_radius'] = torch.zeros(params['means3D'].shape[0]).cuda().float()
        variables['means2D_gradient_accum'] = torch.zeros(params['means3D'].shape[0]).cuda().float()
        variables['denom'] = torch.zeros(params['means3D'].shape[0]).cuda().float()
        variables['timestep'] = torch.zeros(params['means3D'].shape[0]).cuda().float()
        # Load the keyframe time idx list
        keyframe_time_indices = np.load(os.path.join(config['workdir'], config['run_name'], f"keyframe_time_indices{checkpoint_time_idx}.npy"))
        keyframe_time_indices = keyframe_time_indices.tolist()
        # Update the ground truth poses list
        for time_idx in range(checkpoint_time_idx):
            # Load RGBD frames incrementally instead of all frames
            color, depth, _, gt_pose = dataset[time_idx]
            # Process poses
            gt_w2c = torch.linalg.inv(gt_pose)
            gt_w2c_all_frames.append(gt_w2c)
            # Initialize Keyframe List
            if time_idx in keyframe_time_indices:
                # Get the estimated rotation & translation
                curr_cam_rot = F.normalize(params['cam_unnorm_rots'][..., time_idx].detach())
                curr_cam_tran = params['cam_trans'][..., time_idx].detach()
                curr_w2c = torch.eye(4).cuda().float()
                curr_w2c[:3, :3] = build_rotation(curr_cam_rot)
                curr_w2c[:3, 3] = curr_cam_tran
                # Initialize Keyframe Info
                color = color.permute(2, 0, 1) / 255
                depth = depth.permute(2, 0, 1)
                curr_keyframe = {'id': time_idx, 'est_w2c': curr_w2c, 'color': color, 'depth': depth}
                # Add to keyframe list
                keyframe_list.append(curr_keyframe)
    else:
        checkpoint_time_idx = 0
    
    # Iterate over Scan
    if flag_use_semantic and config['model']['flag_use_embedding']==1:
        MLP_func = torch.nn.Conv2d(num_semantic, dataset.num_semantic_class, kernel_size=1)
        MLP_optimizer = torch.optim.Adam(MLP_func.parameters(), lr=5e-4)
        MLP_func.cuda()

    print("[overall] checkpoint_time_idx is: ", checkpoint_time_idx)
    print("[overall] num_frames is: ", num_frames)
    for time_idx in tqdm(range(checkpoint_time_idx, num_frames)):
        
        # Load RGBD frames incrementally instead of all frames
        if not flag_use_semantic:
            color, depth, _, gt_pose = dataset[time_idx]
        else:
            if dataset.use_pyramid:
                color, depth, _, gt_pose, label_gt, pyramid_color, pyramid_depth, pyramid_label_gt = dataset[time_idx]
            else:
                color, depth, _, gt_pose, label_gt = dataset[time_idx]

        
        # Process poses
        gt_w2c = torch.linalg.inv(gt_pose)
        # Process RGB-D Data
        color = color.permute(2, 0, 1) / 255
        depth = depth.permute(2, 0, 1)
        gt_w2c_all_frames.append(gt_w2c)
        curr_gt_w2c = gt_w2c_all_frames
        # Optimize only current time step for tracking
        iter_time_idx = time_idx
        # Initialize Mapping Data for selected frame
        if not flag_use_semantic:
            curr_data = {'cam': cam, 'im': color, 'depth': depth, 'id': iter_time_idx, 'intrinsics': intrinsics, 
                        'w2c': first_frame_w2c, 'iter_gt_w2c_list': curr_gt_w2c}
        else:
            curr_data = {'cam': cam, 'im': color, 'depth': depth, 'id': iter_time_idx, 'intrinsics': intrinsics, 'semantic_gt': label_gt, 
                        'w2c': first_frame_w2c, 'iter_gt_w2c_list': curr_gt_w2c} 
            
        # Initialize Data for Tracking
        if seperate_tracking_res:
            tracking_color, tracking_depth, _, _ = tracking_dataset[time_idx]
            tracking_color = tracking_color.permute(2, 0, 1) / 255
            tracking_depth = tracking_depth.permute(2, 0, 1)
            tracking_curr_data = {'cam': tracking_cam, 'im': tracking_color, 'depth': tracking_depth, 'id': iter_time_idx,
                                  'intrinsics': tracking_intrinsics, 'w2c': first_frame_w2c, 'iter_gt_w2c_list': curr_gt_w2c}
        else:
            tracking_curr_data = curr_data

        # Optimization Iterations
        num_iters_mapping = config['mapping']['num_iters']
        
        # Initialize the camera pose for the current frame
        if time_idx > 0:
            params = initialize_camera_pose(params, time_idx, forward_prop=config['tracking']['forward_prop'])

        # (A). Tracking
        tracking_start_time = time.time()
        if time_idx > 0 and not config['tracking']['use_gt_poses']:
            # Reset Optimizer & Learning Rates for tracking
            optimizer = initialize_optimizer(params, config['tracking']['lrs'], tracking=True)
            # Keep Track of Best Candidate Rotation & Translation
            candidate_cam_unnorm_rot = params['cam_unnorm_rots'][..., time_idx].detach().clone()
            candidate_cam_tran = params['cam_trans'][..., time_idx].detach().clone()
            current_min_loss = float(1e20)
            # Tracking Optimization
            iter = 0
            do_continue_slam = False
            num_iters_tracking = config['tracking']['num_iters']
            tracking_curr_data['iter_tracking'] = iter
            if config['bar']:
                progress_bar = tqdm(range(num_iters_tracking), desc=f"Tracking Time Step: {time_idx}")
            while True:                         # tracking iter
                iter_start_time = time.time()
                # Loss for current frame
                tracking_start_opt_time = time.time()
                if not flag_use_semantic:
                    # without semantic
                    loss, variables, losses = get_loss(params, tracking_curr_data, variables, iter_time_idx, config['tracking']['loss_weights'],
                                                config['tracking']['use_sil_for_loss'], config['tracking']['sil_thres'],
                                                config['tracking']['use_l1'], config['tracking']['ignore_outlier_depth_loss'], tracking=True, 
                                                plot_dir=eval_dir, visualize_tracking_loss=config['tracking']['visualize_tracking_loss'],
                                                tracking_iteration=iter)
                else:
                    # with semantic
                    loss, variables, losses = get_loss_semantic(dataset, params, tracking_curr_data, variables, iter_time_idx, config['tracking']['loss_weights'],
                                                config['tracking']['use_sil_for_loss'], config['tracking']['sil_thres'],
                                                config['tracking']['use_l1'], config['tracking']['ignore_outlier_depth_loss'], vis_label_colorbar, tracking=True, 
                                                plot_dir=eval_dir, visualize_tracking_loss=config['tracking']['visualize_tracking_loss'],
                                                tracking_iteration=iter)
                tracking_end_opt_time = time.time()
                    
                if config['use_wandb']:
                    # Report Loss
                    if flag_use_semantic:
                        wandb_tracking_step = report_loss_semantic(losses, wandb_run, wandb_tracking_step, tracking=True)
                    else:
                        wandb_tracking_step = report_loss(losses, wandb_run, wandb_tracking_step, tracking=True)
                
                loss.backward()
                optimizer.step()

                optimizer.zero_grad(set_to_none=True)
                with torch.no_grad():
                    # Save the best candidate rotation & translation
                    if loss < current_min_loss:
                        current_min_loss = loss
                        candidate_cam_unnorm_rot = params['cam_unnorm_rots'][..., time_idx].detach().clone()
                        candidate_cam_tran = params['cam_trans'][..., time_idx].detach().clone()
                    # Report Progress
                    if config['report_iter_progress']:
                        if config['use_wandb']:
                            report_progress_newrender(params, tracking_curr_data, iter+1, progress_bar, iter_time_idx, sil_thres=config['tracking']['sil_thres'], tracking=True,
                                wandb_run=wandb_run, wandb_step=wandb_tracking_step, 
                                wandb_save_qual=config['wandb']['save_qual'], flag_use_render=1, flag_semantic=flag_use_semantic)
                        else:
                            report_progress(params, tracking_curr_data, iter+1, progress_bar, iter_time_idx, sil_thres=config['tracking']['sil_thres'], tracking=True)
                    else:
                        progress_bar.update(1)
                # Update the runtime numbers
                iter_end_time = time.time()
                tracking_iter_time_sum += iter_end_time - iter_start_time
                tracking_iter_time_count += 1
                
                iter += 1
                if iter == num_iters_tracking:      # tracking end condition
                    if losses['depth'] < config['tracking']['depth_loss_thres'] and config['tracking']['use_depth_loss_thres']:
                        break
                    elif config['tracking']['use_depth_loss_thres'] and not do_continue_slam:
                        do_continue_slam = True
                        progress_bar = tqdm(range(num_iters_tracking), desc=f"Tracking Time Step: {time_idx}")
                        num_iters_tracking = 2*num_iters_tracking
                        if config['use_wandb']:
                            wandb_run.log({"Tracking/Extra Tracking Iters Frames": time_idx,
                                        "Tracking/step": wandb_time_step})
                    else:
                        break

            progress_bar.close()
            # Copy over the best candidate rotation & translation
            with torch.no_grad():
                params['cam_unnorm_rots'][..., time_idx] = candidate_cam_unnorm_rot
                params['cam_trans'][..., time_idx] = candidate_cam_tran
        elif time_idx > 0 and config['tracking']['use_gt_poses']:       # A.2) tracking with GT pose
            with torch.no_grad():
                # Get the ground truth pose relative to frame 0
                rel_w2c = curr_gt_w2c[-1]
                rel_w2c_rot = rel_w2c[:3, :3].unsqueeze(0).detach()
                rel_w2c_rot_quat = matrix_to_quaternion(rel_w2c_rot)
                rel_w2c_tran = rel_w2c[:3, 3].detach()
                # Update the camera parameters
                params['cam_unnorm_rots'][..., time_idx] = rel_w2c_rot_quat
                params['cam_trans'][..., time_idx] = rel_w2c_tran
        # Update the runtime numbers
        tracking_end_time = time.time()
        tracking_frame_time_sum += tracking_end_time - tracking_start_time
        tracking_frame_time_count += 1
        
        if time_idx == 0 or (time_idx+1) % config['report_global_progress_every'] == 0:
            try:
                # Report Final Tracking Progress
                progress_bar = tqdm(range(1), desc=f"Tracking Result Time Step: {time_idx}")
                with torch.no_grad():
                    if config['use_wandb']:
                        report_progress_newrender(params, tracking_curr_data, 1, progress_bar, iter_time_idx, sil_thres=config['tracking']['sil_thres'], tracking=True,
                                wandb_run=wandb_run, wandb_step=wandb_time_step, wandb_save_qual=config['wandb']['save_qual'], 
                                global_logging=True, flag_use_render=1, flag_semantic=flag_use_semantic)
                    else:
                        report_progress(params, tracking_curr_data, 1, progress_bar, iter_time_idx, sil_thres=config['tracking']['sil_thres'], tracking=True)
                progress_bar.close()
            except:
                ckpt_output_dir = os.path.join(config["workdir"], config["run_name"])
                save_params_ckpt(params, ckpt_output_dir, time_idx)
                print('Failed to evaluate trajectory.')

        # (B) Mapping
        # Densification & KeyFrame-based Mapping
        if time_idx == 0 or (time_idx+1) % config['map_every'] == 0:
            # Densification
            if config['mapping']['add_new_gaussians'] and time_idx > 0:
                # Setup Data for Densification
                if seperate_densification_res:
                    # Load RGBD frames incrementally instead of all frames
                    densify_color, densify_depth, _, _ = densify_dataset[time_idx]
                    densify_color = densify_color.permute(2, 0, 1) / 255
                    densify_depth = densify_depth.permute(2, 0, 1)
                    densify_curr_data = {'cam': densify_cam, 'im': densify_color, 'depth': densify_depth, 'id': time_idx, 
                                 'intrinsics': densify_intrinsics, 'w2c': first_frame_w2c, 'iter_gt_w2c_list': curr_gt_w2c}
                else:
                    densify_curr_data = curr_data

                # Add new Gaussians to the scene based on the Silhouette
                if not flag_use_semantic:
                    params, variables = add_new_gaussians_newtest(params, variables, densify_curr_data, 
                                                config['mapping']['sil_thres'], time_idx,
                                                config['mean_sq_dist_method'], config['gaussian_distribution'], flag_use_render=1)
                else:
                    params, variables = add_new_gaussians_semantic_newrender(params, variables, densify_curr_data, 
                                                config['mapping']['sil_thres'], time_idx,
                                                config['mean_sq_dist_method'], num_semantic, flag_use_render=1)
                post_num_pts = params['means3D'].shape[0]
                if config['use_wandb']:
                    wandb_run.log({"Mapping/Number of Gaussians": post_num_pts,
                                   "Mapping/step": wandb_time_step})
            
            with torch.no_grad():
                # Get the current estimated rotation & translation
                curr_cam_rot = F.normalize(params['cam_unnorm_rots'][..., time_idx].detach())
                curr_cam_tran = params['cam_trans'][..., time_idx].detach()
                curr_w2c = torch.eye(4).cuda().float()
                curr_w2c[:3, :3] = build_rotation(curr_cam_rot)
                curr_w2c[:3, 3] = curr_cam_tran
                # Select Keyframes for Mapping
                num_keyframes = config['mapping_window_size']-2
                selected_keyframes = keyframe_selection_overlap(depth, curr_w2c, intrinsics, keyframe_list[:-1], num_keyframes) # every time change?
                selected_time_idx = [keyframe_list[frame_idx]['id'] for frame_idx in selected_keyframes]
                if len(keyframe_list) > 0:
                    # Add last keyframe to the selected keyframes
                    selected_time_idx.append(keyframe_list[-1]['id'])
                    selected_keyframes.append(len(keyframe_list)-1)
                # Add current frame to the selected keyframes
                selected_time_idx.append(time_idx)
                selected_keyframes.append(-1)
                # Print the selected keyframes
                print(f"\nSelected Keyframes at Frame {time_idx}: {selected_time_idx}")

            # Reset Optimizer & Learning Rates for Full Map Optimization
            optimizer = initialize_optimizer(params, config['mapping']['lrs'], tracking=False) 

            # Mapping
            mapping_start_time = time.time()
            if num_iters_mapping > 0:
                progress_bar = tqdm(range(num_iters_mapping), desc=f"Mapping Time Step: {time_idx}")
            # mapping optimization begin ***********
            for iter in range(num_iters_mapping):
                iter_start_time = time.time()
                # Randomly select a frame until current time step amongst keyframes
                rand_idx = np.random.randint(0, len(selected_keyframes))
                selected_rand_keyframe_idx = selected_keyframes[rand_idx]
                if selected_rand_keyframe_idx == -1:
                    # Use Current Frame Data
                    iter_time_idx = time_idx
                    iter_color = color
                    iter_depth = depth
                    if flag_use_semantic:
                        iter_semantic = label_gt
                else:
                    # Use Keyframe Data
                    iter_time_idx = keyframe_list[selected_rand_keyframe_idx]['id']
                    iter_color = keyframe_list[selected_rand_keyframe_idx]['color']
                    iter_depth = keyframe_list[selected_rand_keyframe_idx]['depth']
                    if flag_use_semantic:
                        iter_semantic = keyframe_list[selected_rand_keyframe_idx]['label_gt']
                iter_gt_w2c = gt_w2c_all_frames[:iter_time_idx+1]
                if not flag_use_semantic:
                    iter_data = {'cam': cam, 'im': iter_color, 'depth': iter_depth, 'id': iter_time_idx, 
                                'intrinsics': intrinsics, 'w2c': first_frame_w2c, 'iter_gt_w2c_list': iter_gt_w2c}
                else:
                    iter_data = {'cam': cam, 'im': iter_color, 'depth': iter_depth, 'id': iter_time_idx, 'semantic_label_gt': iter_semantic, 
                             'intrinsics': intrinsics, 'w2c': first_frame_w2c, 'iter_gt_w2c_list': iter_gt_w2c, 'iter_mapping': iter}

                # Loss for current frame
                if not flag_use_semantic:
                    # without semantic (mapping)
                    loss, variables, losses = get_loss(params, iter_data, variables, iter_time_idx, config['mapping']['loss_weights'],
                                                config['mapping']['use_sil_for_loss'], config['mapping']['sil_thres'],
                                                config['mapping']['use_l1'], config['mapping']['ignore_outlier_depth_loss'], mapping=True)
                else:
                    # with semantic (mapping)
                    if config['model']['flag_use_embedding']==0:
                        loss, variables, losses = get_loss_semantic(dataset, params, iter_data, variables, iter_time_idx, config['mapping']['loss_weights'],
                                                    config['mapping']['use_sil_for_loss'], config['mapping']['sil_thres'],
                                                    config['mapping']['use_l1'], config['mapping']['ignore_outlier_depth_loss'], vis_label_colorbar, mapping=True)
                    elif config['model']['flag_use_embedding']==1:
                        loss, variables, losses = get_loss_semantic_mlp(dataset, params, iter_data, variables, iter_time_idx, config['mapping']['loss_weights'],
                                                    config['mapping']['use_sil_for_loss'], config['mapping']['sil_thres'],
                                                    config['mapping']['use_l1'], config['mapping']['ignore_outlier_depth_loss'], vis_label_colorbar, MLP_func, mapping=True)
                
                if config['use_wandb']:
                    # Report Loss
                    if flag_use_semantic:
                        wandb_mapping_step = report_loss_semantic(losses, wandb_run, wandb_mapping_step, mapping=True)
                    else:
                        wandb_mapping_step = report_loss(losses, wandb_run, wandb_mapping_step, mapping=True)
                # Backprop
                loss.backward()
                with torch.no_grad():
                    # Prune Gaussians
                    if config['mapping']['prune_gaussians']:
                        params, variables = prune_gaussians(params, variables, optimizer, iter, config['mapping']['pruning_dict'])
                        if config['use_wandb']:
                            wandb_run.log({"Mapping/Number of Gaussians - Pruning": params['means3D'].shape[0],
                                           "Mapping/step": wandb_mapping_step})
                    # Gaussian-Splatting's Gradient-based Densification
                    if config['mapping']['use_gaussian_splatting_densification']:
                        params, variables = densify(params, variables, optimizer, iter, config['mapping']['densify_dict'])
                        if config['use_wandb']:
                            wandb_run.log({"Mapping/Number of Gaussians - Densification": params['means3D'].shape[0],
                                           "Mapping/step": wandb_mapping_step})
                            
                            
                    optimizer.step()
                    if flag_use_semantic and config['model']['flag_use_embedding']==1:
                        MLP_optimizer.step()

                    optimizer.zero_grad(set_to_none=True)
                    if flag_use_semantic and config['model']['flag_use_embedding']==1:
                        MLP_optimizer.zero_grad()

                    # Report Progress
                    if config['report_iter_progress']:
                        if config['use_wandb']:
                            report_progress_newrender(params, iter_data, iter+1, progress_bar, iter_time_idx, sil_thres=config['mapping']['sil_thres'], 
                                            wandb_run=wandb_run, wandb_step=wandb_mapping_step, wandb_save_qual=config['wandb']['save_qual'],
                                            mapping=True, online_time_idx=time_idx, flag_use_render=1, flag_semantic=flag_use_semantic)
                        else:
                            report_progress_newrender(params, iter_data, iter+1, progress_bar, iter_time_idx, sil_thres=config['mapping']['sil_thres'], 
                                            mapping=True, online_time_idx=time_idx, flag_semantic=flag_use_semantic)
                    else:
                        progress_bar.update(1)
                # Update the runtime numbers
                iter_end_time = time.time()
                mapping_iter_time_sum += iter_end_time - iter_start_time
                mapping_iter_time_count += 1


            if num_iters_mapping > 0:
                progress_bar.close()
            # Update the runtime numbers
            mapping_end_time = time.time()
            mapping_frame_time_sum += mapping_end_time - mapping_start_time
            mapping_frame_time_count += 1
            # print("\nmapping/frame time is: {:.4f}s\n".format(mapping_end_time - mapping_start_time))

            if time_idx == 0 or (time_idx+1) % config['report_global_progress_every'] == 0:
                try:
                    # Report Mapping Progress
                    progress_bar = tqdm(range(1), desc=f"Mapping Result Time Step: {time_idx}")
                    with torch.no_grad():
                        if config['use_wandb']:
                            report_progress_newrender(params, curr_data, 1, progress_bar, time_idx, sil_thres=config['mapping']['sil_thres'], 
                                    wandb_run=wandb_run, wandb_step=wandb_time_step, wandb_save_qual=config['wandb']['save_qual'],
                                    mapping=True, online_time_idx=time_idx, global_logging=True, flag_use_render=1, flag_semantic=flag_use_semantic)
                        else:
                            report_progress(params, curr_data, 1, progress_bar, time_idx, sil_thres=config['mapping']['sil_thres'], 
                                    mapping=True, online_time_idx=time_idx)
                    progress_bar.close()
                except:
                    ckpt_output_dir = os.path.join(config["workdir"], config["run_name"])
                    save_params_ckpt(params, ckpt_output_dir, time_idx)
                    if flag_use_semantic and config['model']['flag_use_embedding']:    # save mlp
                        torch.save(MLP_func.state_dict(), os.path.join(ckpt_output_dir, "Semantic_{}".format(time_idx)+'.pth'))
                    print('Failed to evaluate trajectory.')
        

        # Keyframe Selection
        if ((time_idx == 0) or ((time_idx+1) % config['keyframe_every'] == 0) or \
                    (time_idx == num_frames-2)) and (not torch.isinf(curr_gt_w2c[-1]).any()) and (not torch.isnan(curr_gt_w2c[-1]).any()):
            with torch.no_grad():
                # Get the current estimated rotation & translation
                curr_cam_rot = F.normalize(params['cam_unnorm_rots'][..., time_idx].detach())
                curr_cam_tran = params['cam_trans'][..., time_idx].detach()
                curr_w2c = torch.eye(4).cuda().float()
                curr_w2c[:3, :3] = build_rotation(curr_cam_rot)
                curr_w2c[:3, 3] = curr_cam_tran
                # Initialize Keyframe Info
                if flag_use_semantic:
                    curr_keyframe = {'id': time_idx, 'est_w2c': curr_w2c, 'color': color, 'depth': depth, 'label_gt': label_gt, 'cam': cam, 'intrinsics': intrinsics}
                else:
                    curr_keyframe = {'id': time_idx, 'est_w2c': curr_w2c, 'color': color, 'depth': depth, 'cam': cam, 'intrinsics': intrinsics}
                # Add to keyframe list
                keyframe_list.append(curr_keyframe)
                keyframe_time_indices.append(time_idx)

                
        # Checkpoint every iteration
        if time_idx % config["checkpoint_interval"] == 0 and config['save_checkpoints']:
            ckpt_output_dir = os.path.join(config["workdir"], config["run_name"])
            save_params_ckpt(params, ckpt_output_dir, time_idx)
            np.save(os.path.join(ckpt_output_dir, f"keyframe_time_indices{time_idx}.npy"), np.array(keyframe_time_indices))
            if flag_use_semantic and config['model']['flag_use_embedding']:    # save mlp
                torch.save(MLP_func.state_dict(), os.path.join(ckpt_output_dir, "Semantic_{}".format(time_idx)+'.pth'))
        
        # Increment WandB Time Step
        if config['use_wandb']:
            wandb_time_step += 1

        torch.cuda.empty_cache()

    # Compute Average Runtimes
    if tracking_iter_time_count == 0:
        tracking_iter_time_count = 1
        tracking_frame_time_count = 1
    if mapping_iter_time_count == 0:
        mapping_iter_time_count = 1
        mapping_frame_time_count = 1
    tracking_iter_time_avg = tracking_iter_time_sum / tracking_iter_time_count
    tracking_frame_time_avg = tracking_frame_time_sum / tracking_frame_time_count
    mapping_iter_time_avg = mapping_iter_time_sum / mapping_iter_time_count
    mapping_frame_time_avg = mapping_frame_time_sum / mapping_frame_time_count
    print(f"\nAverage Tracking/Iteration Time: {tracking_iter_time_avg*1000} ms")
    print(f"Average Tracking/Frame Time: {tracking_frame_time_avg} s")
    print(f"Average Mapping/Iteration Time: {mapping_iter_time_avg*1000} ms")
    print(f"Average Mapping/Frame Time: {mapping_frame_time_avg} s")
    if config['use_wandb']:
        wandb_run.log({"Final Stats/Average Tracking Iteration Time (ms)": tracking_iter_time_avg*1000,
                       "Final Stats/Average Tracking Frame Time (s)": tracking_frame_time_avg,
                       "Final Stats/Average Mapping Iteration Time (ms)": mapping_iter_time_avg*1000,
                       "Final Stats/Average Mapping Frame Time (s)": mapping_frame_time_avg,
                       "Final Stats/step": 1})
        
    # Add Camera Parameters to Save them
    params['timestep'] = variables['timestep']
    params['intrinsics'] = intrinsics.detach().cpu().numpy()
    params['w2c'] = first_frame_w2c.detach().cpu().numpy()
    params['org_width'] = dataset_config["desired_image_width"]
    params['org_height'] = dataset_config["desired_image_height"]
    params['gt_w2c_all_frames'] = []
    for gt_w2c_tensor in gt_w2c_all_frames:
        params['gt_w2c_all_frames'].append(gt_w2c_tensor.detach().cpu().numpy())
    params['gt_w2c_all_frames'] = np.stack(params['gt_w2c_all_frames'], axis=0)
    params['keyframe_time_indices'] = np.array(keyframe_time_indices)
    
    # Save Parameters
    save_params(params, output_dir)
    if flag_use_semantic and config['model']['flag_use_embedding']:    # save mlp
        torch.save(MLP_func.state_dict(), os.path.join(output_dir, "Semantic.pth"))
    
    # Evaluate Final Parameters
    with torch.no_grad():
        if config['use_wandb']:
            if not flag_use_semantic:
                # no semantic pipeline (eval)
                eval_newrender(dataset, params, num_frames, eval_dir, sil_thres=config['mapping']['sil_thres'],
                                wandb_run=wandb_run, wandb_save_qual=config['wandb']['eval_save_qual'],
                                mapping_iters=config['mapping']['num_iters'], add_new_gaussians=config['mapping']['add_new_gaussians'],
                                eval_every=config['eval_every'], flag_use_render = 1)
            else:
                # semantic pipeline (eval)
                if dataset.sem_mode == "binary1" or dataset.sem_mode == "eigen13" or dataset.sem_mode == "nyu40" or dataset.sem_mode == "raw":
                    # flat semantic representation
                    eval_semantic_newrender(dataset, params, num_frames, eval_dir, sil_thres=config['mapping']['sil_thres'],
                        wandb_run=wandb_run, wandb_save_qual=config['wandb']['eval_save_qual'],
                        mapping_iters=config['mapping']['num_iters'], add_new_gaussians=config['mapping']['add_new_gaussians'],
                        eval_every=config['eval_every'])
                else:
                    # tree semantic representation
                    if config['model']['flag_use_embedding'] == 1:
                        eval_semantic_tree_newrender(dataset, params, num_frames, eval_dir, sil_thres=config['mapping']['sil_thres'],
                        wandb_run=wandb_run, wandb_save_qual=config['wandb']['eval_save_qual'],
                        mapping_iters=config['mapping']['num_iters'], add_new_gaussians=config['mapping']['add_new_gaussians'],
                        eval_every=config['eval_every'], save_frames = True, flag_use_embedding=config['model']['flag_use_embedding'], mlp_func=MLP_func, gt_transfer=config['model']['eval_gt_transfer'])
                    else:
                        eval_semantic_tree_newrender(dataset, params, num_frames, eval_dir, sil_thres=config['mapping']['sil_thres'],
                        wandb_run=wandb_run, wandb_save_qual=config['wandb']['eval_save_qual'],
                        mapping_iters=config['mapping']['num_iters'], add_new_gaussians=config['mapping']['add_new_gaussians'],
                        eval_every=config['eval_every'], save_frames = True)
        else:
            # not use wandb
            if not flag_use_semantic:
                # no semantic pipeline (eval)
                eval_newrender(dataset, params, num_frames, eval_dir, sil_thres=config['mapping']['sil_thres'],
                            mapping_iters=config['mapping']['num_iters'], add_new_gaussians=config['mapping']['add_new_gaussians'],
                            eval_every=config['eval_every'], flag_use_render=1)
            else:
                # semantic pipeline (eval)
                if dataset.sem_mode == "original":
                    # flat semantic representation
                    eval_semantic_newrender(dataset, params, num_frames, eval_dir, sil_thres=config['mapping']['sil_thres'],
                                mapping_iters=config['mapping']['num_iters'], add_new_gaussians=config['mapping']['add_new_gaussians'],
                                eval_every=config['eval_every'], save_frames=True) 
                elif "tree" in dataset.sem_mode:
                    # tree semantic representation
                    if config['model']['flag_use_embedding'] == 1:
                        eval_semantic_tree_newrender(dataset, params, num_frames, eval_dir, sil_thres=config['mapping']['sil_thres'],
                        mapping_iters=config['mapping']['num_iters'], add_new_gaussians=config['mapping']['add_new_gaussians'],
                        eval_every=config['eval_every'], save_frames = False, flag_mlp = config['model']['flag_use_embedding'], mlp_func=MLP_func)
                    else:
                        eval_semantic_tree_newrender(dataset, params, num_frames, eval_dir, sil_thres=config['mapping']['sil_thres'],
                        mapping_iters=config['mapping']['num_iters'], add_new_gaussians=config['mapping']['add_new_gaussians'],
                        eval_every=config['eval_every'], save_frames = True)


    # Close WandB Run
    if config['use_wandb']:
        wandb.finish()

if __name__ == "__main__":
    start_time = time.time()
    
    parser = argparse.ArgumentParser()

    parser.add_argument("experiment", type=str, help="Path to experiment file")

    args = parser.parse_args()

    experiment = SourceFileLoader(
        os.path.basename(args.experiment), args.experiment
    ).load_module()

    # Set Experiment Seed
    seed_everything(seed=experiment.config['seed'])
    
    # Create Results Directory and Copy Config
    results_dir = os.path.join(
        experiment.config["workdir"], experiment.config["run_name"]
    )
    experiment.config["results_dir"] = results_dir
    if not experiment.config['load_checkpoint']:
        os.makedirs(results_dir, exist_ok=True)
        shutil.copy(args.experiment, os.path.join(results_dir, "config.py"))

    hierslam_main(experiment.config)

    end_time = time.time()
    time_splatam = end_time-start_time
    print('the whole splatam time use is: {}s, {}min, {}h.'.format(time_splatam, time_splatam/60.00, time_splatam/3600.00))