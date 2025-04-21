import cv2
import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import time
from imgviz import label_colormap
import imageio

from datasets.gradslam_datasets.geometryutils import relative_transformation
from utils.recon_helpers import setup_camera
from utils.slam_external import build_rotation, calc_psnr
from utils.slam_helpers import (
    transform_to_frame, transformed_params2rendervar, transformed_params2rendervar_semantic, transformed_params2depthplussilhouette,
    quat_mult, matrix_to_quaternion
)

from diff_gaussian_rasterization import GaussianRasterizer as Renderer
from diff_gaussian_rasterization import GaussianRasterizer_semantic as Renderer_semantic

from pytorch_msssim import ms_ssim
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
loss_fn_alex = LearnedPerceptualImagePatchSimilarity(net_type='alex', normalize=True).cuda()

class_name_string_nyu40 = ["void","wall", "floor", "cabinet", "bed", "chair",
                    "sofa", "table", "door", "window", "bookshelf", 
                    "picture", "counter", "blinds", "desk", "shelves",
                    "curtain", "dresser", "pillow", "mirror", "floor",
                    "clothes", "ceiling", "books", "fridge", "tv",
                    "paper", "towel", "shower curtain", "box", "white board",
                    "person", "night stand", "toilet", "sink", "lamp",
                    "bath tub", "bag", "other struct", "other furniture", "other prop"]

def mask_to_boundary(mask, dilation_ratio=0.02):
    """
    Convert binary mask to boundary mask.
    :param mask (numpy array, uint8): binary mask
    :param dilation_ratio (float): ratio to calculate dilation = dilation_ratio * image_diagonal
    :return: boundary mask (numpy array)
    """
    h, w = mask.shape
    img_diag = np.sqrt(h ** 2 + w ** 2)
    dilation = int(round(dilation_ratio * img_diag))
    if dilation < 1:
        dilation = 1
    # Pad image so mask truncated by the image border is also considered as boundary.
    new_mask = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    kernel = np.ones((3, 3), dtype=np.uint8)
    new_mask_erode = cv2.erode(new_mask, kernel, iterations=dilation)
    mask_erode = new_mask_erode[1 : h + 1, 1 : w + 1]
    # G_d intersects G in the paper.
    return mask - mask_erode

def boundary_iou(gt, dt, dilation_ratio=0.02):
    """
    Compute boundary iou between two binary masks.
    :param gt (numpy array, uint8): binary mask
    :param dt (numpy array, uint8): binary mask
    :param dilation_ratio (float): ratio to calculate dilation = dilation_ratio * image_diagonal
    :return: boundary iou (float)
    """
    dt = (dt>0.0).astype('uint8')
    gt = (gt>0.0).astype('uint8')
    
    gt_boundary = mask_to_boundary(gt, dilation_ratio)
    dt_boundary = mask_to_boundary(dt, dilation_ratio)
    
    # Save gt_boundary to a file
    if False:
        gt_boundary[gt_boundary == 1] = 255
        dt_boundary[dt_boundary == 1] = 255
        cv2.imwrite('./gt_boundary.png', gt_boundary)
        cv2.imwrite('./dt_boundary.png', dt_boundary)

    intersection = ((gt_boundary * dt_boundary) > 0).sum()
    union = ((gt_boundary + dt_boundary) > 0).sum()
    boundary_iou = intersection / union
    return boundary_iou

def calculate_iou(mask1, mask2):
    """Calculate IoU between two boolean masks."""
    mask1_bool = mask1 > 0.0
    mask2_bool = mask2 > 0.0
    intersection = np.logical_and(mask1_bool, mask2_bool)
    union = np.logical_or(mask1_bool, mask2_bool)   
    iou = np.sum(intersection) / np.sum(union)
    return iou

def find_positions(matrix, key_find):
    matrix_reshaped = matrix.permute(1, 2, 0)
    mask = key_find != -1
    submatrix = matrix_reshaped[:, :, mask]
    subkey = key_find[mask]
    matches = (submatrix == subkey).all(dim=-1)
    return matches

def semantic_label_vis(label_map_tree, colour_map_level):
    """
    label_map_tree:   [level_num, h, w]
    colour_map_level: [h, w, 3]
    """

    label_map_tree_vis = torch.zeros(label_map_tree.shape[1], label_map_tree.shape[2], 3, dtype=torch.uint8)
    for key, value in colour_map_level.items():
        # print(key, value)
        key = key.unsqueeze(1)
        key = key.unsqueeze(1)
        res = (label_map_tree==key)
        res = torch.all(res, dim=0)
        color_value = torch.from_numpy(value)
        label_map_tree_vis[res] = color_value
    label_map_tree_vis = np.array(label_map_tree_vis)
    
    return label_map_tree_vis

def semantic_label_vis_replica(label_map_tree, colour_map_level, iter=-1):
    """
    label_map_tree:   [level_num, h, w]
    colour_map_level: [h, w, 3]
    """

    label_map_tree_vis = torch.zeros(label_map_tree.shape[1], label_map_tree.shape[2], 3, dtype=torch.uint8)
    for key, value in colour_map_level.items():
        res = find_positions(label_map_tree, key)
        color_value = torch.from_numpy(value)
        label_map_tree_vis[res] = color_value
        
    label_map_tree_vis = np.array(label_map_tree_vis)

    return label_map_tree_vis

def transfer_tree_2_label(dataset, im_semantic_treelabel, flag_keystr = False):
    """
    transfer 3-level label map -> base label map
    """
    im_semantic_label = torch.full((im_semantic_treelabel.shape[1], im_semantic_treelabel.shape[2]), -1)

    if not im_semantic_treelabel.is_cuda:
        im_semantic_treelabel = im_semantic_treelabel.to('cuda')
    
    for key, value in dataset.label_mapping_tree.items():
        value = torch.tensor(value).unsqueeze(1)
        value = value.unsqueeze(2).to('cuda')
        index = (im_semantic_treelabel==value)      # [3, h, w]
        index = torch.all(index, dim=0)
        if flag_keystr:
            key = int(key)
        im_semantic_label[index] = key

    indices_nolabel = torch.where(im_semantic_label == -1)
    print("\n len is: ", len(indices_nolabel[0]))

    return im_semantic_label

def transfer_eachlevel_1(im_semantic_levels, dataset):
    level_num = im_semantic_levels.shape[0]
    idx_mapping = dataset.tree_id_classes_map[level_num-1]
    num_dim = len(idx_mapping)
    im_semantic_label = torch.full((im_semantic_levels.shape[1], im_semantic_levels.shape[2]), -1)

    for idx, (key, value) in enumerate(idx_mapping.items(), start=0):
        key = torch.tensor(key).unsqueeze(1)
        key = key.unsqueeze(2)
        index = (im_semantic_levels==key)      # [level_num, h, w] Fix: Comparing with the unsqueezed key tensor
        index = torch.all(index, dim=0)
        im_semantic_label[index] = idx

    return im_semantic_label, num_dim

def transfer_back_eachlevel(im_semantic, dataset, i_level):
    im_semantic = im_semantic.unsqueeze(0)
    idx_mapping = dataset.label_mapping_tree
    im_semantic_label = torch.full((im_semantic.shape[1], im_semantic.shape[2]), -1)

    for idx, (key, value) in enumerate(idx_mapping.items(), start=0):
        key = int(key)
        key = torch.tensor(key).unsqueeze(0).unsqueeze(1).unsqueeze(2)
        index = (im_semantic==key)      # [level_num, h, w] Fix: Comparing with the unsqueezed key tensor
        index = torch.all(index, dim=0)
        im_semantic_label[index] = value[i_level]

    return im_semantic_label

def transfer_tree_label(im_semantic, tree_num_semantic):
    logits_2_label = lambda x: torch.argmax(torch.nn.functional.softmax(x, dim=-1),dim=-1)
        
    idx_begin = 0
    for i in range(len(tree_num_semantic)-1):
        num_classes_level = tree_num_semantic[i]
        idx_end = idx_begin + num_classes_level
        im_semantic_level = im_semantic[idx_begin:idx_end, :, :]
        semantic_label = logits_2_label(im_semantic_level.permute(1,2,0))
        semantic_label = semantic_label.unsqueeze(0)
        
        if i == 0:
            im_semantic_treelabel = semantic_label
        else:
            im_semantic_treelabel = torch.cat((im_semantic_treelabel, semantic_label), dim=0)
        idx_begin = idx_end
    return im_semantic_treelabel

# visualization
def visualize_label(label_mask, colour_map, flag_save=True, save_name="none"):
    if flag_save:
        if isinstance(label_mask, torch.Tensor):
            label_mask = label_mask.cpu().numpy().astype(int)  # Convert tensor to numpy array
        elif isinstance(label_mask, np.ndarray):
            label_mask = label_mask.astype(int)
        label_vis = label_mask * 255
        label_vis = label_vis.astype(np.uint8)
        label_vis = np.squeeze(label_vis)
        print("===> save at: ", save_name)
        cv2.imwrite(save_name + ".png", label_vis)

def align(model, data):
    """Align two trajectories using the method of Horn (closed-form).

    Args:
        model -- first trajectory (3xn)
        data -- second trajectory (3xn)

    Returns:
        rot -- rotation matrix (3x3)
        trans -- translation vector (3x1)
        trans_error -- translational error per point (1xn)

    """
    np.set_printoptions(precision=3, suppress=True)
    model_zerocentered = model - model.mean(1).reshape((3,-1))
    data_zerocentered = data - data.mean(1).reshape((3,-1))

    W = np.zeros((3, 3))
    for column in range(model.shape[1]):
        W += np.outer(model_zerocentered[:,
                         column], data_zerocentered[:, column])
    U, d, Vh = np.linalg.linalg.svd(W.transpose())
    S = np.matrix(np.identity(3))
    if (np.linalg.det(U) * np.linalg.det(Vh) < 0):
        S[2, 2] = -1
    rot = U*S*Vh
    trans = data.mean(1).reshape((3,-1)) - rot * model.mean(1).reshape((3,-1))

    model_aligned = rot * model + trans
    alignment_error = model_aligned - data

    trans_error = np.sqrt(np.sum(np.multiply(
        alignment_error, alignment_error), 0)).A[0]

    return rot, trans, trans_error

def evaluate_ate(gt_traj, est_traj):
    """
    Input : 
        gt_traj: list of 4x4 matrices 
        est_traj: list of 4x4 matrices
        len(gt_traj) == len(est_traj)
    """
    flag_debug = False

    gt_traj_pts = [gt_traj[idx][:3,3] for idx in range(len(gt_traj))]
    est_traj_pts = [est_traj[idx][:3,3] for idx in range(len(est_traj))]

    gt_traj_pts  = torch.stack(gt_traj_pts).detach().cpu().numpy().T
    est_traj_pts = torch.stack(est_traj_pts).detach().cpu().numpy().T

    _, _, trans_error = align(gt_traj_pts, est_traj_pts)

    avg_trans_error = trans_error.mean()
    if flag_debug:
        print("trans_error is: \n", trans_error)

    return avg_trans_error

def report_loss(losses, wandb_run, wandb_step, tracking=False, mapping=False):
    # Update loss dict
    loss_dict = {'Loss': losses['loss'].item(),
                 'Image Loss': losses['im'].item(),
                 'Depth Loss': losses['depth'].item(),}
    if tracking:
        tracking_loss_dict = {}
        for k, v in loss_dict.items():
            tracking_loss_dict[f"Per Iteration Tracking/{k}"] = v
        tracking_loss_dict['Per Iteration Tracking/step'] = wandb_step
        wandb_run.log(tracking_loss_dict)
    elif mapping:
        mapping_loss_dict = {}
        for k, v in loss_dict.items():
            mapping_loss_dict[f"Per Iteration Mapping/{k}"] = v
        mapping_loss_dict['Per Iteration Mapping/step'] = wandb_step
        wandb_run.log(mapping_loss_dict)
    else:
        frame_opt_loss_dict = {}
        for k, v in loss_dict.items():
            frame_opt_loss_dict[f"Per Iteration Current Frame Optimization/{k}"] = v
        frame_opt_loss_dict['Per Iteration Current Frame Optimization/step'] = wandb_step
        wandb_run.log(frame_opt_loss_dict)
    
    # Increment wandb step
    wandb_step += 1
    return wandb_step
        
def report_loss_semantic(losses, wandb_run, wandb_step, tracking=False, mapping=False):
    # Update loss dict
    if tracking:    # tracking
        loss_dict = {'Loss': losses['loss'].item(),
                 'Image Loss': losses['im'].item(),
                 'Depth Loss': losses['depth'].item()
                    }  
    else:   # mapping
        loss_dict = {'Loss': losses['loss'].item(),
                 'Image Loss': losses['im'].item(),
                 'Depth Loss': losses['depth'].item(),
                 'Semantic Loss': losses['sem'].item(),
                    }
    if tracking:
        tracking_loss_dict = {}
        for k, v in loss_dict.items():
            tracking_loss_dict[f"Per Iteration Tracking/{k}"] = v
        tracking_loss_dict['Per Iteration Tracking/step'] = wandb_step
        wandb_run.log(tracking_loss_dict)
    elif mapping:
        mapping_loss_dict = {}
        for k, v in loss_dict.items():
            mapping_loss_dict[f"Per Iteration Mapping/{k}"] = v
        mapping_loss_dict['Per Iteration Mapping/step'] = wandb_step
        wandb_run.log(mapping_loss_dict)
    else:
        frame_opt_loss_dict = {}
        for k, v in loss_dict.items():
            frame_opt_loss_dict[f"Per Iteration Current Frame Optimization/{k}"] = v
        frame_opt_loss_dict['Per Iteration Current Frame Optimization/step'] = wandb_step
        wandb_run.log(frame_opt_loss_dict)
    
    # Increment wandb step
    wandb_step += 1
    return wandb_step

def plot_rgbd_silhouette(color, depth, rastered_color, rastered_depth, presence_sil_mask, diff_depth_l1,
                         psnr, depth_l1, fig_title, plot_dir=None, plot_name=None, 
                         save_plot=False, wandb_run=None, wandb_step=None, wandb_title=None, diff_rgb=None):
    # Determine Plot Aspect Ratio
    aspect_ratio = color.shape[2] / color.shape[1]
    fig_height = 8
    fig_width = 14/1.55
    fig_width = fig_width * aspect_ratio
    # Plot the Ground Truth and Rasterized RGB & Depth, along with Diff Depth & Silhouette
    fig, axs = plt.subplots(2, 3, figsize=(fig_width, fig_height))
    axs[0, 0].imshow(color.cpu().permute(1, 2, 0))
    axs[0, 0].set_title("Ground Truth RGB")
    axs[0, 1].imshow(depth[0, :, :].cpu(), cmap='jet', vmin=0, vmax=6)
    axs[0, 1].set_title("Ground Truth Depth")
    rastered_color = torch.clamp(rastered_color, 0, 1)
    axs[1, 0].imshow(rastered_color.cpu().permute(1, 2, 0))
    axs[1, 0].set_title("Rasterized RGB, PSNR: {:.2f}".format(psnr))
    axs[1, 1].imshow(rastered_depth[0, :, :].cpu(), cmap='jet', vmin=0, vmax=6)
    axs[1, 1].set_title("Rasterized Depth, L1: {:.2f}".format(depth_l1))
    if diff_rgb is not None:
        axs[0, 2].imshow(diff_rgb.cpu(), cmap='jet', vmin=0, vmax=6)
        axs[0, 2].set_title("Diff RGB L1")
    else:
        axs[0, 2].imshow(presence_sil_mask, cmap='gray')
        axs[0, 2].set_title("Rasterized Silhouette")
    diff_depth_l1 = diff_depth_l1.cpu().squeeze(0)
    axs[1, 2].imshow(diff_depth_l1, cmap='jet', vmin=0, vmax=6)
    axs[1, 2].set_title("Diff Depth L1")
    for ax in axs.flatten():
        ax.axis('off')
    fig.suptitle(fig_title, y=0.95, fontsize=16)
    fig.tight_layout()
    if save_plot:
        save_path = os.path.join(plot_dir, f"{plot_name}.png")
        plt.savefig(save_path, bbox_inches='tight')
    if wandb_run is not None:
        if wandb_step is None:
            wandb_run.log({wandb_title: fig})
        else:
            wandb_run.log({wandb_title: fig}, step=wandb_step)
    plt.close()

def report_progress(params, data, i, progress_bar, iter_time_idx, sil_thres, every_i=1, qual_every_i=1, 
                    tracking=False, mapping=False, wandb_run=None, wandb_step=None, wandb_save_qual=False, online_time_idx=None,
                    global_logging=True):
    if i % every_i == 0 or i == 1:
        if wandb_run is not None:
            if tracking:
                stage = "Tracking"
            elif mapping:
                stage = "Mapping"
            else:
                stage = "Current Frame Optimization"
        if not global_logging:
            stage = "Per Iteration " + stage

        if tracking:
            # Get list of gt poses
            gt_w2c_list = data['iter_gt_w2c_list']
            valid_gt_w2c_list = []
            
            # Get latest trajectory
            latest_est_w2c = data['w2c']
            latest_est_w2c_list = []
            latest_est_w2c_list.append(latest_est_w2c)
            valid_gt_w2c_list.append(gt_w2c_list[0])
            for idx in range(1, iter_time_idx+1):
                # Check if gt pose is not nan for this time step
                if torch.isnan(gt_w2c_list[idx]).sum() > 0:
                    continue
                interm_cam_rot = F.normalize(params['cam_unnorm_rots'][..., idx].detach())
                interm_cam_trans = params['cam_trans'][..., idx].detach()
                intermrel_w2c = torch.eye(4).cuda().float()
                intermrel_w2c[:3, :3] = build_rotation(interm_cam_rot)
                intermrel_w2c[:3, 3] = interm_cam_trans
                latest_est_w2c = intermrel_w2c
                latest_est_w2c_list.append(latest_est_w2c)
                valid_gt_w2c_list.append(gt_w2c_list[idx])

            # Get latest gt pose
            gt_w2c_list = valid_gt_w2c_list
            iter_gt_w2c = gt_w2c_list[-1]
            # Get euclidean distance error between latest and gt pose
            iter_pt_error = torch.sqrt((latest_est_w2c[0,3] - iter_gt_w2c[0,3])**2 + (latest_est_w2c[1,3] - iter_gt_w2c[1,3])**2 + (latest_est_w2c[2,3] - iter_gt_w2c[2,3])**2)
            if iter_time_idx > 0:
                # Calculate relative pose error
                rel_gt_w2c = relative_transformation(gt_w2c_list[-2], gt_w2c_list[-1])
                rel_est_w2c = relative_transformation(latest_est_w2c_list[-2], latest_est_w2c_list[-1])
                rel_pt_error = torch.sqrt((rel_gt_w2c[0,3] - rel_est_w2c[0,3])**2 + (rel_gt_w2c[1,3] - rel_est_w2c[1,3])**2 + (rel_gt_w2c[2,3] - rel_est_w2c[2,3])**2)
            else:
                rel_pt_error = torch.zeros(1).float()
            
            # Calculate ATE RMSE
            ate_rmse = evaluate_ate(gt_w2c_list, latest_est_w2c_list)
            ate_rmse = np.round(ate_rmse, decimals=6)
            if wandb_run is not None:
                tracking_log = {f"{stage}/Latest Pose Error":iter_pt_error, 
                               f"{stage}/Latest Relative Pose Error":rel_pt_error,
                               f"{stage}/ATE RMSE":ate_rmse}

        # Get current frame Gaussians
        transformed_gaussians = transform_to_frame(params, iter_time_idx, 
                                                   gaussians_grad=False,
                                                   camera_grad=False)

        # Initialize Render Variables
        rendervar = transformed_params2rendervar(params, transformed_gaussians)
        depth_sil_rendervar = transformed_params2depthplussilhouette(params, data['w2c'], 
                                                                     transformed_gaussians)
        depth_sil, _, _, = Renderer(raster_settings=data['cam'])(**depth_sil_rendervar)
        rastered_depth = depth_sil[0, :, :].unsqueeze(0)
        valid_depth_mask = (data['depth'] > 0)
        silhouette = depth_sil[1, :, :]
        presence_sil_mask = (silhouette > sil_thres)

        im, _, _, = Renderer(raster_settings=data['cam'])(**rendervar)
        if tracking:
            psnr = calc_psnr(im * presence_sil_mask, data['im'] * presence_sil_mask).mean()
        else:
            psnr = calc_psnr(im, data['im']).mean()

        if tracking:
            diff_depth_rmse = torch.sqrt((((rastered_depth - data['depth']) * presence_sil_mask) ** 2))
            diff_depth_rmse = diff_depth_rmse * valid_depth_mask
            rmse = diff_depth_rmse.sum() / valid_depth_mask.sum()
            diff_depth_l1 = torch.abs((rastered_depth - data['depth']) * presence_sil_mask)
            diff_depth_l1 = diff_depth_l1 * valid_depth_mask
            depth_l1 = diff_depth_l1.sum() / valid_depth_mask.sum()
        else:
            diff_depth_rmse = torch.sqrt((((rastered_depth - data['depth'])) ** 2))
            diff_depth_rmse = diff_depth_rmse * valid_depth_mask
            rmse = diff_depth_rmse.sum() / valid_depth_mask.sum()
            diff_depth_l1 = torch.abs((rastered_depth - data['depth']))
            diff_depth_l1 = diff_depth_l1 * valid_depth_mask
            depth_l1 = diff_depth_l1.sum() / valid_depth_mask.sum()

        if not (tracking or mapping):
            progress_bar.set_postfix({f"Time-Step: {iter_time_idx} | PSNR: {psnr:.{7}} | Depth RMSE: {rmse:.{7}} | L1": f"{depth_l1:.{7}}"})
            progress_bar.update(every_i)
        elif tracking:
            progress_bar.set_postfix({f"Time-Step: {iter_time_idx} | Rel Pose Error: {rel_pt_error.item():.{7}} | Pose Error: {iter_pt_error.item():.{7}} | ATE RMSE": f"{ate_rmse.item():.{7}}"})
            progress_bar.update(every_i)
        elif mapping:
            progress_bar.set_postfix({f"Time-Step: {online_time_idx} | Frame {data['id']} | PSNR: {psnr:.{7}} | Depth RMSE: {rmse:.{7}} | L1": f"{depth_l1:.{7}}"})
            progress_bar.update(every_i)
        
        if wandb_run is not None:
            wandb_log = {f"{stage}/PSNR": psnr,
                         f"{stage}/Depth RMSE": rmse,
                         f"{stage}/Depth L1": depth_l1,
                         f"{stage}/step": wandb_step}
            if tracking:
                wandb_log = {**wandb_log, **tracking_log}
            wandb_run.log(wandb_log)
        
        if wandb_save_qual and (i % qual_every_i == 0 or i == 1):
            # Silhouette Mask
            presence_sil_mask = presence_sil_mask.detach().cpu().numpy()

            # Log plot to wandb
            if not mapping:
                fig_title = f"Time-Step: {iter_time_idx} | Iter: {i} | Frame: {data['id']}"
            else:
                fig_title = f"Time-Step: {online_time_idx} | Iter: {i} | Frame: {data['id']}"
            plot_rgbd_silhouette(data['im'], data['depth'], im, rastered_depth, presence_sil_mask, diff_depth_l1,
                                 psnr, depth_l1, fig_title, wandb_run=wandb_run, wandb_step=wandb_step, 
                                 wandb_title=f"{stage} Qual Viz")

def report_progress_newrender(params, data, i, progress_bar, iter_time_idx, sil_thres, every_i=1, qual_every_i=1, 
                    tracking=False, mapping=False, wandb_run=None, wandb_step=None, wandb_save_qual=False, online_time_idx=None,
                    global_logging=True, flag_use_render=1, flag_semantic=True):
    if i % every_i == 0 or i == 1:
        if wandb_run is not None:
            if tracking:
                stage = "Tracking"
            elif mapping:
                stage = "Mapping"
            else:
                stage = "Current Frame Optimization"
        if not global_logging:
            stage = "Per Iteration " + stage

        if tracking:
            # Get list of gt poses
            gt_w2c_list = data['iter_gt_w2c_list']
            valid_gt_w2c_list = []
            
            # Get latest trajectory
            latest_est_w2c = data['w2c']
            latest_est_w2c_list = []
            latest_est_w2c_list.append(latest_est_w2c)
            valid_gt_w2c_list.append(gt_w2c_list[0])
            for idx in range(1, iter_time_idx+1):
                # Check if gt pose is not nan for this time step
                if torch.isnan(gt_w2c_list[idx]).sum() > 0:
                    continue
                interm_cam_rot = F.normalize(params['cam_unnorm_rots'][..., idx].detach())
                interm_cam_trans = params['cam_trans'][..., idx].detach()
                intermrel_w2c = torch.eye(4).cuda().float()
                intermrel_w2c[:3, :3] = build_rotation(interm_cam_rot)
                intermrel_w2c[:3, 3] = interm_cam_trans
                latest_est_w2c = intermrel_w2c
                latest_est_w2c_list.append(latest_est_w2c)
                valid_gt_w2c_list.append(gt_w2c_list[idx])

            # Get latest gt pose
            gt_w2c_list = valid_gt_w2c_list
            iter_gt_w2c = gt_w2c_list[-1]
            # Get euclidean distance error between latest and gt pose
            iter_pt_error = torch.sqrt((latest_est_w2c[0,3] - iter_gt_w2c[0,3])**2 + (latest_est_w2c[1,3] - iter_gt_w2c[1,3])**2 + (latest_est_w2c[2,3] - iter_gt_w2c[2,3])**2)
            if iter_time_idx > 0:
                # Calculate relative pose error
                rel_gt_w2c = relative_transformation(gt_w2c_list[-2], gt_w2c_list[-1])
                rel_est_w2c = relative_transformation(latest_est_w2c_list[-2], latest_est_w2c_list[-1])
                rel_pt_error = torch.sqrt((rel_gt_w2c[0,3] - rel_est_w2c[0,3])**2 + (rel_gt_w2c[1,3] - rel_est_w2c[1,3])**2 + (rel_gt_w2c[2,3] - rel_est_w2c[2,3])**2)
            else:
                rel_pt_error = torch.zeros(1).float()
            
            # Calculate ATE RMSE
            ate_rmse = evaluate_ate(gt_w2c_list, latest_est_w2c_list)
            ate_rmse = np.round(ate_rmse, decimals=6)
            if wandb_run is not None:
                tracking_log = {f"{stage}/Latest Pose Error":iter_pt_error, 
                               f"{stage}/Latest Relative Pose Error":rel_pt_error,
                               f"{stage}/ATE RMSE":ate_rmse}

        # Get current frame Gaussians
        transformed_gaussians = transform_to_frame(params, iter_time_idx, 
                                                   gaussians_grad=False,
                                                   camera_grad=False)

        # Initialize Render Variables
        if not flag_semantic:
            rendervar = transformed_params2rendervar(params, transformed_gaussians)
        else:
            rendervar_sem = transformed_params2rendervar_semantic(params, transformed_gaussians)
        
        if flag_use_render == 1 and not flag_semantic:      
            # no semantic
            im, radius, rendered_depth, rendered_median_depth, rendered_final_opcity, rendered_mask = \
            Renderer(raster_settings=data['cam'])(**rendervar)
        elif flag_use_render == 1 and flag_semantic:
            # with semantic
            im, radius, im_semantic, rendered_depth, rendered_median_depth, rendered_final_opcity = \
            Renderer_semantic(raster_settings=data['cam'])(**rendervar_sem)
    
        rastered_depth = rendered_depth
        valid_depth_mask = (data['depth'] > 0)
        # mask (use rendered_final_opcity) / rendered_mask
        silhouette = rendered_final_opcity
        presence_sil_mask = (silhouette > sil_thres)

        if tracking:
            psnr = calc_psnr(im * presence_sil_mask, data['im'] * presence_sil_mask).mean()
        else:
            psnr = calc_psnr(im, data['im']).mean()

        if tracking:
            diff_depth_rmse = torch.sqrt((((rastered_depth - data['depth']) * presence_sil_mask) ** 2))
            diff_depth_rmse = diff_depth_rmse * valid_depth_mask
            rmse = diff_depth_rmse.sum() / valid_depth_mask.sum()
            diff_depth_l1 = torch.abs((rastered_depth - data['depth']) * presence_sil_mask)
            diff_depth_l1 = diff_depth_l1 * valid_depth_mask
            depth_l1 = diff_depth_l1.sum() / valid_depth_mask.sum()
        else:
            diff_depth_rmse = torch.sqrt((((rastered_depth - data['depth'])) ** 2))
            diff_depth_rmse = diff_depth_rmse * valid_depth_mask
            rmse = diff_depth_rmse.sum() / valid_depth_mask.sum()
            diff_depth_l1 = torch.abs((rastered_depth - data['depth']))
            diff_depth_l1 = diff_depth_l1 * valid_depth_mask
            depth_l1 = diff_depth_l1.sum() / valid_depth_mask.sum()

        if not (tracking or mapping):
            progress_bar.set_postfix({f"Time-Step: {iter_time_idx} | PSNR: {psnr:.{7}} | Depth RMSE: {rmse:.{7}} | L1": f"{depth_l1:.{7}}"})
            progress_bar.update(every_i)
        elif tracking:
            progress_bar.set_postfix({f"Time-Step: {iter_time_idx} | Rel Pose Error: {rel_pt_error.item():.{7}} | Pose Error: {iter_pt_error.item():.{7}} | ATE RMSE": f"{ate_rmse.item():.{7}}"})
            progress_bar.update(every_i)
        elif mapping:
            progress_bar.set_postfix({f"Time-Step: {online_time_idx} | Frame {data['id']} | PSNR: {psnr:.{7}} | Depth RMSE: {rmse:.{7}} | L1": f"{depth_l1:.{7}}"})
            progress_bar.update(every_i)
        
        if wandb_run is not None:
            wandb_log = {f"{stage}/PSNR": psnr,
                         f"{stage}/Depth RMSE": rmse,
                         f"{stage}/Depth L1": depth_l1,
                         f"{stage}/step": wandb_step}
            if tracking:
                wandb_log = {**wandb_log, **tracking_log}
            wandb_run.log(wandb_log)
        
        if wandb_save_qual and (i % qual_every_i == 0 or i == 1):
            # Silhouette Mask
            presence_sil_mask = presence_sil_mask.detach().cpu().numpy()

            # Log plot to wandb
            if not mapping:
                fig_title = f"Time-Step: {iter_time_idx} | Iter: {i} | Frame: {data['id']}"
            else:
                fig_title = f"Time-Step: {online_time_idx} | Iter: {i} | Frame: {data['id']}"
            plot_rgbd_silhouette(data['im'], data['depth'], im, rastered_depth, presence_sil_mask, diff_depth_l1,
                                 psnr, depth_l1, fig_title, wandb_run=wandb_run, wandb_step=wandb_step, 
                                 wandb_title=f"{stage} Qual Viz")

def eval_newrender(dataset, final_params, num_frames, eval_dir, sil_thres, 
         mapping_iters, add_new_gaussians, wandb_run=None, wandb_save_qual=False, eval_every=1, save_frames=False, flag_use_render=1):
    print("Evaluating Final Parameters ...")
    psnr_list = []
    rmse_list = []
    l1_list = []
    lpips_list = []
    ssim_list = []
    plot_dir = os.path.join(eval_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    if save_frames:
        render_rgb_dir = os.path.join(eval_dir, "rendered_rgb")
        os.makedirs(render_rgb_dir, exist_ok=True)
        render_depth_dir = os.path.join(eval_dir, "rendered_depth")
        os.makedirs(render_depth_dir, exist_ok=True)
        rgb_dir = os.path.join(eval_dir, "rgb")
        os.makedirs(rgb_dir, exist_ok=True)
        depth_dir = os.path.join(eval_dir, "depth")
        os.makedirs(depth_dir, exist_ok=True)

    gt_w2c_list = []
    for time_idx in tqdm(range(num_frames)):
         # Get RGB-D Data & Camera Parameters
        color, depth, intrinsics, pose = dataset[time_idx]
        gt_w2c = torch.linalg.inv(pose)
        gt_w2c_list.append(gt_w2c)
        intrinsics = intrinsics[:3, :3]

        # Process RGB-D Data
        color = color.permute(2, 0, 1) / 255 # (H, W, C) -> (C, H, W)
        depth = depth.permute(2, 0, 1) # (H, W, C) -> (C, H, W)

        if time_idx == 0:
            # Process Camera Parameters
            first_frame_w2c = torch.linalg.inv(pose)
            # Setup Camera
            cam = setup_camera(color.shape[2], color.shape[1], intrinsics.cpu().numpy(), first_frame_w2c.detach().cpu().numpy())
        
        # Skip frames if not eval_every
        if time_idx != 0 and (time_idx+1) % eval_every != 0:
            continue
        print("eval frame: ", time_idx)

        # Get current frame Gaussians
        transformed_gaussians = transform_to_frame(final_params, time_idx, 
                                                   gaussians_grad=False, 
                                                   camera_grad=False)
 
        # Define current frame data
        curr_data = {'cam': cam, 'im': color, 'depth': depth, 'id': time_idx, 'intrinsics': intrinsics, 'w2c': first_frame_w2c}

        # Initialize Render Variables
        rendervar = transformed_params2rendervar(final_params, transformed_gaussians)

        # Render Depth & Silhouette & img
        if flag_use_render == 1:
            im, radius, rendered_depth, rendered_median_depth, rendered_final_opcity, rendered_mask = \
            Renderer(raster_settings=curr_data['cam'])(**rendervar)
        elif flag_use_render == 2:
            im, radius, rendered_depth, rendered_final_opcity = \
            Renderer(raster_settings=curr_data['cam'])(**rendervar)

        rastered_depth = rendered_depth
        # Mask invalid depth in GT
        valid_depth_mask = (curr_data['depth'] > 0)
        rastered_depth_viz = rastered_depth.detach()
        rastered_depth = rastered_depth * valid_depth_mask
        silhouette = rendered_final_opcity.squeeze(0)
        presence_sil_mask = (silhouette > sil_thres)

        if mapping_iters==0 and not add_new_gaussians:
            weighted_im = im * presence_sil_mask * valid_depth_mask
            weighted_gt_im = curr_data['im'] * presence_sil_mask * valid_depth_mask
        else:
            weighted_im = im * valid_depth_mask
            weighted_gt_im = curr_data['im'] * valid_depth_mask
        psnr = calc_psnr(weighted_im, weighted_gt_im).mean()
        ssim = ms_ssim(weighted_im.unsqueeze(0).cpu(), weighted_gt_im.unsqueeze(0).cpu(), 
                        data_range=1.0, size_average=True)
        lpips_score = loss_fn_alex(torch.clamp(weighted_im.unsqueeze(0), 0.0, 1.0),
                                    torch.clamp(weighted_gt_im.unsqueeze(0), 0.0, 1.0)).item()

        psnr_list.append(psnr.cpu().numpy())
        ssim_list.append(ssim.cpu().numpy())
        lpips_list.append(lpips_score)

        # Compute Depth RMSE
        if mapping_iters==0 and not add_new_gaussians:
            diff_depth_rmse = torch.sqrt((((rastered_depth - curr_data['depth']) * presence_sil_mask) ** 2))
            diff_depth_rmse = diff_depth_rmse * valid_depth_mask
            rmse = diff_depth_rmse.sum() / valid_depth_mask.sum()
            diff_depth_l1 = torch.abs((rastered_depth - curr_data['depth']) * presence_sil_mask)
            diff_depth_l1 = diff_depth_l1 * valid_depth_mask
            depth_l1 = diff_depth_l1.sum() / valid_depth_mask.sum()
        else:
            diff_depth_rmse = torch.sqrt((((rastered_depth - curr_data['depth'])) ** 2))
            diff_depth_rmse = diff_depth_rmse * valid_depth_mask
            rmse = diff_depth_rmse.sum() / valid_depth_mask.sum()
            diff_depth_l1 = torch.abs((rastered_depth - curr_data['depth']))
            diff_depth_l1 = diff_depth_l1 * valid_depth_mask
            depth_l1 = diff_depth_l1.sum() / valid_depth_mask.sum()
        rmse_list.append(rmse.cpu().numpy())
        l1_list.append(depth_l1.cpu().numpy())

        if save_frames:
            # Save Rendered RGB and Depth
            viz_render_im = torch.clamp(im, 0, 1)
            viz_render_im = viz_render_im.detach().cpu().permute(1, 2, 0).numpy()
            vmin = 0
            vmax = 6
            viz_render_depth = rastered_depth_viz[0].detach().cpu().numpy()
            normalized_depth = np.clip((viz_render_depth - vmin) / (vmax - vmin), 0, 1)
            depth_colormap = cv2.applyColorMap((normalized_depth * 255).astype(np.uint8), cv2.COLORMAP_JET)
            cv2.imwrite(os.path.join(render_rgb_dir, "gs_{:04d}.png".format(time_idx)), cv2.cvtColor(viz_render_im*255, cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(render_depth_dir, "gs_{:04d}.png".format(time_idx)), depth_colormap)

            # Save GT RGB and Depth
            viz_gt_im = torch.clamp(curr_data['im'], 0, 1)
            viz_gt_im = viz_gt_im.detach().cpu().permute(1, 2, 0).numpy()
            viz_gt_depth = curr_data['depth'][0].detach().cpu().numpy()
            normalized_depth = np.clip((viz_gt_depth - vmin) / (vmax - vmin), 0, 1)
            depth_colormap = cv2.applyColorMap((normalized_depth * 255).astype(np.uint8), cv2.COLORMAP_JET)
            cv2.imwrite(os.path.join(rgb_dir, "gt_{:04d}.png".format(time_idx)), cv2.cvtColor(viz_gt_im*255, cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(depth_dir, "gt_{:04d}.png".format(time_idx)), depth_colormap)
        
        # Plot the Ground Truth and Rasterized RGB & Depth, along with Silhouette
        fig_title = "Time Step: {}".format(time_idx)
        plot_name = "%04d" % time_idx
        presence_sil_mask = presence_sil_mask.detach().cpu().numpy()
        if wandb_run is None:
            plot_rgbd_silhouette(color, depth, im, rastered_depth_viz, presence_sil_mask, diff_depth_l1,
                                 psnr, depth_l1, fig_title, plot_dir, 
                                 plot_name=plot_name, save_plot=True)
        elif wandb_save_qual:
            plot_rgbd_silhouette(color, depth, im, rastered_depth_viz, presence_sil_mask, diff_depth_l1,
                                 psnr, depth_l1, fig_title, plot_dir, 
                                 plot_name=plot_name, save_plot=True,
                                 wandb_run=wandb_run, wandb_step=None, 
                                 wandb_title="Eval/Qual Viz")

    try:
        # Compute the final ATE RMSE
        # Get the final camera trajectory
        num_frames = final_params['cam_unnorm_rots'].shape[-1]
        latest_est_w2c = first_frame_w2c
        latest_est_w2c_list = []
        latest_est_w2c_list.append(latest_est_w2c)
        valid_gt_w2c_list = []
        valid_gt_w2c_list.append(gt_w2c_list[0])
        for idx in range(1, num_frames):
            # Check if gt pose is not nan for this time step
            if torch.isnan(gt_w2c_list[idx]).sum() > 0:
                continue
            interm_cam_rot = F.normalize(final_params['cam_unnorm_rots'][..., idx].detach())
            interm_cam_trans = final_params['cam_trans'][..., idx].detach()
            intermrel_w2c = torch.eye(4).cuda().float()
            intermrel_w2c[:3, :3] = build_rotation(interm_cam_rot)
            intermrel_w2c[:3, 3] = interm_cam_trans
            latest_est_w2c = intermrel_w2c
            latest_est_w2c_list.append(latest_est_w2c)
            valid_gt_w2c_list.append(gt_w2c_list[idx])
        gt_w2c_list = valid_gt_w2c_list
        # Calculate ATE RMSE
        ate_rmse = evaluate_ate(gt_w2c_list, latest_est_w2c_list)
        print("Final Average ATE RMSE: {:.2f} cm".format(ate_rmse*100))
        if wandb_run is not None:
            wandb_run.log({"Final Stats/Avg ATE RMSE": ate_rmse,
                        "Final Stats/step": 1})
    except:
        ate_rmse = 100.0
        print('Failed to evaluate trajectory with alignment.')
    
    # Compute Average Metrics
    psnr_list = np.array(psnr_list)
    rmse_list = np.array(rmse_list)
    l1_list = np.array(l1_list)
    ssim_list = np.array(ssim_list)
    lpips_list = np.array(lpips_list)
    avg_psnr = psnr_list.mean()
    avg_rmse = rmse_list.mean()
    avg_l1 = l1_list.mean()
    avg_ssim = ssim_list.mean()
    avg_lpips = lpips_list.mean()
    print("Average PSNR: {:.2f}".format(avg_psnr))
    print("Average MS-SSIM: {:.3f}".format(avg_ssim))
    print("Average LPIPS: {:.3f}".format(avg_lpips))
    print("Average Depth RMSE: {:.2f} cm".format(avg_rmse*100))
    print("Average Depth L1: {:.2f} cm".format(avg_l1*100))
    
    print(" ==== summary ==== ")
    print("[ATE RMSE] [PSNR] [MS-SSIM] [LPIPS] [Depth L1] [Depth RMSE]")
    print("{:.2f}\t{:.2f}\t{:.3f}\t{:.3f}\t{:.2f}\t{:.2f}".format(ate_rmse*100, avg_psnr, avg_ssim, avg_lpips, avg_l1*100, avg_rmse*100))

    if wandb_run is not None:
        wandb_run.log({"Final Stats/Average PSNR": avg_psnr, 
                       "Final Stats/Average Depth RMSE": avg_rmse,
                       "Final Stats/Average Depth L1": avg_l1,
                       "Final Stats/Average MS-SSIM": avg_ssim, 
                       "Final Stats/Average LPIPS": avg_lpips,
                       "Final Stats/step": 1})

    # Save metric lists as text files
    np.savetxt(os.path.join(eval_dir, "psnr.txt"), psnr_list)
    np.savetxt(os.path.join(eval_dir, "rmse.txt"), rmse_list)
    np.savetxt(os.path.join(eval_dir, "l1.txt"), l1_list)
    np.savetxt(os.path.join(eval_dir, "ssim.txt"), ssim_list)
    np.savetxt(os.path.join(eval_dir, "lpips.txt"), lpips_list)

    # Plot PSNR & L1 as line plots
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    axs[0].plot(np.arange(len(psnr_list)), psnr_list)
    axs[0].set_title("RGB PSNR")
    axs[0].set_xlabel("Time Step")
    axs[0].set_ylabel("PSNR")
    axs[1].plot(np.arange(len(l1_list)), l1_list*100)
    axs[1].set_title("Depth L1")
    axs[1].set_xlabel("Time Step")
    axs[1].set_ylabel("L1 (cm)")
    fig.suptitle("Average PSNR: {:.2f}, Average Depth L1: {:.2f} cm, ATE RMSE: {:.2f} cm".format(avg_psnr, avg_l1*100, ate_rmse*100), y=1.05, fontsize=16)
    plt.savefig(os.path.join(eval_dir, "metrics.png"), bbox_inches='tight')
    if wandb_run is not None:
        wandb_run.log({"Eval/Metrics": fig})
    plt.close()

def eval_semantic_newrender(dataset, final_params, num_frames, eval_dir, sil_thres, 
         mapping_iters, add_new_gaussians, wandb_run=None, 
         wandb_save_qual=False, eval_every=1, save_frames=False):
    
    print("Evaluating Final Parameters ...")
    print("using Renderer_semantic")
    psnr_list = []
    rmse_list = []
    l1_list = []
    lpips_list = []
    ssim_list = []
    miou_list = []
    mbiou_list = []
    plot_dir = os.path.join(eval_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    render_rgb_dir = os.path.join(eval_dir, "rendered_rgb")
    os.makedirs(render_rgb_dir, exist_ok=True)
    render_depth_dir = os.path.join(eval_dir, "rendered_depth")
    os.makedirs(render_depth_dir, exist_ok=True)
    render_sem_dir = os.path.join(eval_dir, "rendered_semantic")
    os.makedirs(render_sem_dir, exist_ok=True)
    rgb_dir = os.path.join(eval_dir, "rgb")
    os.makedirs(rgb_dir, exist_ok=True)
    depth_dir = os.path.join(eval_dir, "depth")
    os.makedirs(depth_dir, exist_ok=True)

    gt_w2c_list = []
    for time_idx in tqdm(range(num_frames)):
        # Get RGB-D Data & Camera Parameters
        color, depth, intrinsics, pose, label_gt = dataset[time_idx]
        gt_w2c = torch.linalg.inv(pose)
        gt_w2c_list.append(gt_w2c)
        intrinsics = intrinsics[:3, :3]

        # Process RGB-D Data
        color = color.permute(2, 0, 1) / 255 # (H, W, C) -> (C, H, W)
        depth = depth.permute(2, 0, 1) # (H, W, C) -> (C, H, W)

        if time_idx == 0:
            # Process Camera Parameters
            first_frame_w2c = torch.linalg.inv(pose)
            # Setup Camera
            cam = setup_camera(color.shape[2], color.shape[1], intrinsics.cpu().numpy(), first_frame_w2c.detach().cpu().numpy())
        
        # Skip frames if not eval_every 
        if time_idx != 0 and (time_idx+1) % eval_every != 0:
            continue

        # Get current frame Gaussians
        transformed_pts = transform_to_frame(final_params, time_idx, 
                                             gaussians_grad=False,
                                             camera_grad=False)
 
        # Define current frame data
        curr_data = {'cam': cam, 'im': color, 'depth': depth, 'id': time_idx, 'intrinsics': intrinsics, 'w2c': first_frame_w2c, 'label_gt':label_gt}

        # Initialize Render Variables
        rendervar_sem = transformed_params2rendervar_semantic(final_params, transformed_pts)

        # render
        im, radius, im_semantic, rendered_depth, rendered_median_depth, rendered_final_opcity =\
            Renderer_semantic(raster_settings=curr_data['cam'])(**rendervar_sem)
        
        rastered_depth = rendered_depth
        valid_depth_mask = (curr_data['depth'] > 0)
        rastered_depth_viz = rastered_depth.detach()
        rastered_depth = rastered_depth * valid_depth_mask
        silhouette = rendered_final_opcity.squeeze(0)
        presence_sil_mask = (silhouette > sil_thres)
        
        if mapping_iters==0 and not add_new_gaussians:
            weighted_im = im * presence_sil_mask * valid_depth_mask
            weighted_gt_im = curr_data['im'] * presence_sil_mask * valid_depth_mask
        else:
            weighted_im = im * valid_depth_mask
            weighted_gt_im = curr_data['im'] * valid_depth_mask
        psnr = calc_psnr(weighted_im, weighted_gt_im).mean()
        ssim = ms_ssim(weighted_im.unsqueeze(0).cpu(), weighted_gt_im.unsqueeze(0).cpu(), 
                        data_range=1.0, size_average=True)
        lpips_score = loss_fn_alex(torch.clamp(weighted_im.unsqueeze(0), 0.0, 1.0),
                                    torch.clamp(weighted_gt_im.unsqueeze(0), 0.0, 1.0)).item()

        psnr_list.append(psnr.cpu().numpy())
        ssim_list.append(ssim.cpu().numpy())
        lpips_list.append(lpips_score)

        # Compute Depth RMSE
        if mapping_iters==0 and not add_new_gaussians:
            diff_depth_rmse = torch.sqrt((((rastered_depth - curr_data['depth']) * presence_sil_mask) ** 2))
            diff_depth_rmse = diff_depth_rmse * valid_depth_mask
            rmse = diff_depth_rmse.sum() / valid_depth_mask.sum()
            diff_depth_l1 = torch.abs((rastered_depth - curr_data['depth']) * presence_sil_mask)
            diff_depth_l1 = diff_depth_l1 * valid_depth_mask
            depth_l1 = diff_depth_l1.sum() / valid_depth_mask.sum()
        else:
            diff_depth_rmse = torch.sqrt((((rastered_depth - curr_data['depth'])) ** 2))
            diff_depth_rmse = diff_depth_rmse * valid_depth_mask
            rmse = diff_depth_rmse.sum() / valid_depth_mask.sum()
            diff_depth_l1 = torch.abs((rastered_depth - curr_data['depth']))
            diff_depth_l1 = diff_depth_l1 * valid_depth_mask
            depth_l1 = diff_depth_l1.sum() / valid_depth_mask.sum()
        rmse_list.append(rmse.cpu().numpy())
        l1_list.append(depth_l1.cpu().numpy())

        ## semantic loss
        logits_2_label = lambda x: torch.argmax(torch.nn.functional.softmax(x, dim=-1),dim=-1)
        
        start_time = time.time()
        iou_scores = {}  # Store IoU scores for each class
        biou_scores = {}
        class_counts = {}  # Count the number of times each class appears
        semantic_label = logits_2_label(im_semantic.permute(1,2,0))
        semantic_label_gt = curr_data['label_gt'].squeeze(-1)
        
        # each semantic class
        flag_printsem = True
        if flag_printsem:
            print("current frame is: {}".format(time_idx))
        for sem_label in range(dataset.num_semantic_class):
            # transform label to binary mask
            sem_label_mask = (semantic_label == sem_label).float()
            sem_label_gt_mask = (semantic_label_gt == sem_label).float()
            if sem_label_mask.sum() == 0 and sem_label_gt_mask.sum() == 0:
                continue
            
            sem_label_mask = sem_label_mask.cpu().numpy()
            sem_label_gt_mask = sem_label_gt_mask.cpu().numpy()
            # visualize (optional)
            visualize_label(sem_label_mask, dataset.colour_map_np, flag_save = False, save_name = os.path.join(render_sem_dir, "sem_{:04d}_class{}".format(time_idx, sem_label)) )
            visualize_label(sem_label_gt_mask, dataset.colour_map_np, flag_save = False, save_name = os.path.join(render_sem_dir, "gt_{:04d}_class{}".format(time_idx, sem_label)) )
            # compute iou
            iou = calculate_iou(sem_label_gt_mask, sem_label_mask)
            biou = boundary_iou(sem_label_gt_mask, sem_label_mask)
            # accumulate iou
            if sem_label not in iou_scores:
                iou_scores[sem_label] = []
                biou_scores[sem_label] = []
            iou_scores[sem_label].append(iou)
            biou_scores[sem_label].append(biou)
            class_counts[sem_label] = class_counts.get(sem_label, 0) + 1
            if flag_printsem:
                if dataset.num_semantic_class==41:
                    print(" semantic label {} ({}): iou: {:.3f}, biou: {:.3f}, class_counts: {}".format(sem_label, class_name_string_nyu40[sem_label], iou, biou, class_counts[sem_label]))
                else:
                    print(" semantic label {} ({}): iou: {:.3f}, biou: {:.3f}, class_counts: {}".format(sem_label, dataset.semantic_class[sem_label], iou, biou, class_counts[sem_label]))
        
        # Calculate mean IoU for each class
        mean_iou_per_class = {cat_id: np.mean(iou_scores[cat_id]) for cat_id in iou_scores}
        mean_biou_per_class = {cat_id: np.mean(biou_scores[cat_id]) for cat_id in biou_scores}

        # Calculate overall mean IoU
        overall_mean_iou = np.mean(list(mean_iou_per_class.values()))
        overall_mean_biou = np.mean(list(mean_biou_per_class.values()))

        if flag_printsem:
            print("mean_iou: {:.4f}, mean_biou: {:.4f}".format(overall_mean_iou, overall_mean_biou))

        miou_list.append(overall_mean_iou)
        mbiou_list.append(overall_mean_biou)

        if False:
            print("Mean IoU per class:", mean_iou_per_class)
            print("Mean Boundary IoU per class:", mean_biou_per_class)
            print("Overall Mean IoU:", overall_mean_iou)
            print("Overall Boundary Mean IoU:", overall_mean_biou)
        
        if save_frames:
            # Save Rendered RGB and Depth
            viz_render_im = torch.clamp(im, 0, 1)
            viz_render_im = viz_render_im.detach().cpu().permute(1, 2, 0).numpy()
            vmin = 0
            vmax = 6
            viz_render_depth = rastered_depth_viz[0].detach().cpu().numpy()
            normalized_depth = np.clip((viz_render_depth - vmin) / (vmax - vmin), 0, 1)
            depth_colormap = cv2.applyColorMap((normalized_depth * 255).astype(np.uint8), cv2.COLORMAP_JET)
            cv2.imwrite(os.path.join(render_rgb_dir, "gs_{:04d}.png".format(time_idx)), cv2.cvtColor(viz_render_im*255, cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(render_depth_dir, "gs_{:04d}.png".format(time_idx)), depth_colormap)

            # Save Rendered im_semantic
            logits_2_label = lambda x: torch.argmax(torch.nn.functional.softmax(x, dim=-1),dim=-1)
            if True:
                im_semantic = im_semantic.permute(1,2,0)
                semantic_map_label = logits_2_label(im_semantic)
                semantic_map_label = semantic_map_label.detach().cpu().numpy()  # [480, 640]
                print(semantic_map_label.max())     # 20
                semantic_map_label_colorbar = dataset.colour_map_np[semantic_map_label] # [14, 3]
                semantic_map_label_colorbar = semantic_map_label_colorbar.astype(np.uint8) 
                semantic_map_label_colorbar = np.squeeze(semantic_map_label_colorbar) 
                cv2.imwrite(os.path.join(render_sem_dir, "sem_{:04d}.png".format(time_idx) ), semantic_map_label_colorbar[...,::-1]) # bgr -> rgb  

            # Save GT RGB and Depth
            viz_gt_im = torch.clamp(curr_data['im'], 0, 1)
            viz_gt_im = viz_gt_im.detach().cpu().permute(1, 2, 0).numpy()
            viz_gt_depth = curr_data['depth'][0].detach().cpu().numpy()
            normalized_depth = np.clip((viz_gt_depth - vmin) / (vmax - vmin), 0, 1)
            depth_colormap = cv2.applyColorMap((normalized_depth * 255).astype(np.uint8), cv2.COLORMAP_JET)
            cv2.imwrite(os.path.join(rgb_dir, "gt_{:04d}.png".format(time_idx)), cv2.cvtColor(viz_gt_im*255, cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(depth_dir, "gt_{:04d}.png".format(time_idx)), depth_colormap)
            
            # semantic visualization
            label_gt = label_gt.detach().cpu().numpy().astype(int)
            vis_semantic_gt_label = dataset.colour_map_np[label_gt]
            label_gt_vis = vis_semantic_gt_label.astype(np.uint8) 
            label_gt_vis = np.squeeze(label_gt_vis)
            cv2.imwrite(os.path.join(render_sem_dir, "gt_{:04d}.png".format(time_idx)), label_gt_vis[...,::-1]) # bgr -> rgb  
        
        # Plot the Ground Truth and Rasterized RGB & Depth, along with Silhouette
        fig_title = "Time Step: {}".format(time_idx)
        plot_name = "%04d" % time_idx
        presence_sil_mask = presence_sil_mask.detach().cpu().numpy()
        if wandb_run is None:
            plot_rgbd_silhouette(color, depth, im, rastered_depth_viz, presence_sil_mask, diff_depth_l1,
                                 psnr, depth_l1, fig_title, plot_dir, 
                                 plot_name=plot_name, save_plot=True)
        elif wandb_save_qual:
            plot_rgbd_silhouette(color, depth, im, rastered_depth_viz, presence_sil_mask, diff_depth_l1,
                                 psnr, depth_l1, fig_title, plot_dir, 
                                 plot_name=plot_name, save_plot=True,
                                 wandb_run=wandb_run, wandb_step=None, 
                                 wandb_title="Eval/Qual Viz")

    try:
        # Compute the final ATE RMSE
        # Get the final camera trajectory
        num_frames = final_params['cam_unnorm_rots'].shape[-1]
        latest_est_w2c = first_frame_w2c
        latest_est_w2c_list = []
        latest_est_w2c_list.append(latest_est_w2c)
        valid_gt_w2c_list = []
        valid_gt_w2c_list.append(gt_w2c_list[0])
        for idx in range(1, num_frames):
            # Check if gt pose is not nan for this time step
            if torch.isnan(gt_w2c_list[idx]).sum() > 0:
                continue
            interm_cam_rot = F.normalize(final_params['cam_unnorm_rots'][..., idx].detach())
            interm_cam_trans = final_params['cam_trans'][..., idx].detach()
            intermrel_w2c = torch.eye(4).cuda().float()
            intermrel_w2c[:3, :3] = build_rotation(interm_cam_rot)
            intermrel_w2c[:3, 3] = interm_cam_trans
            latest_est_w2c = intermrel_w2c
            latest_est_w2c_list.append(latest_est_w2c)
            valid_gt_w2c_list.append(gt_w2c_list[idx])
        gt_w2c_list = valid_gt_w2c_list
        # Calculate ATE RMSE
        ate_rmse = evaluate_ate(gt_w2c_list, latest_est_w2c_list)
        print("Final Average ATE RMSE: {:.2f} cm".format(ate_rmse*100))
        if wandb_run is not None:
            wandb_run.log({"Final Stats/Avg ATE RMSE": ate_rmse,
                        "Final Stats/step": 1})
    except:
        ate_rmse = 100.0
        print('Failed to evaluate trajectory with alignment.')
    
    # Compute Average Metrics
    psnr_list = np.array(psnr_list)
    rmse_list = np.array(rmse_list)
    l1_list = np.array(l1_list)
    ssim_list = np.array(ssim_list)
    lpips_list = np.array(lpips_list)
    miou_list = np.array(miou_list)
    mbiou_list = np.array(mbiou_list)
    avg_psnr = psnr_list.mean()
    avg_rmse = rmse_list.mean()
    avg_l1 = l1_list.mean()
    avg_ssim = ssim_list.mean()
    avg_lpips = lpips_list.mean()
    avg_miou = miou_list.mean() 
    avg_mbiou = mbiou_list.mean()
    print("Average PSNR: {:.2f}".format(avg_psnr))
    print("Average MS-SSIM: {:.3f}".format(avg_ssim))
    print("Average LPIPS: {:.3f}".format(avg_lpips))
    print("Average Depth L1: {:.2f} cm".format(avg_l1*100))
    print("Average Depth RMSE: {:.2f} cm".format(avg_rmse*100))
    print("Average miou: {:.3f}".format(avg_miou))
    print("Average mbiou: {:.3f}".format(avg_mbiou))
    print(" ==== summary ==== ")
    print("[ATE RMSE] [PSNR] [MS-SSIM] [LPIPS] [Depth L1] [Depth RMSE] [miou] [mbiou]")
    print("{:.2f}\t{:.2f}\t{:.3f}\t{:.3f}\t{:.2f}\t{:.2f}\t{:.3f}\t{:.3f}".format(ate_rmse*100, avg_psnr, avg_ssim, avg_lpips, avg_l1*100, avg_rmse*100, avg_miou, avg_mbiou))


    if wandb_run is not None:
        wandb_run.log({"Final Stats/Average PSNR": avg_psnr, 
                       "Final Stats/Average MS-SSIM": avg_ssim, 
                       "Final Stats/Average LPIPS": avg_lpips,
                       "Final Stats/Average Depth L1": avg_l1,
                       "Final Stats/Average Depth RMSE": avg_rmse,
                       "Final Stats/semantic miou": avg_miou,
                       "Final Stats/semantic mbiou": avg_mbiou,
                       "Final Stats/step": 1})

    # Save metric lists as text files
    np.savetxt(os.path.join(eval_dir, "psnr.txt"), psnr_list)
    np.savetxt(os.path.join(eval_dir, "rmse.txt"), rmse_list)
    np.savetxt(os.path.join(eval_dir, "l1.txt"), l1_list)
    np.savetxt(os.path.join(eval_dir, "ssim.txt"), ssim_list)
    np.savetxt(os.path.join(eval_dir, "lpips.txt"), lpips_list)
    np.savetxt(os.path.join(eval_dir, "miou.txt"), miou_list)       
    np.savetxt(os.path.join(eval_dir, "mbiou.txt"), mbiou_list)     

    # Plot PSNR & L1 as line plots
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    axs[0].plot(np.arange(len(psnr_list)), psnr_list)
    axs[0].set_title("RGB PSNR")
    axs[0].set_xlabel("Time Step")
    axs[0].set_ylabel("PSNR")
    axs[1].plot(np.arange(len(l1_list)), l1_list*100)
    axs[1].set_title("Depth L1")
    axs[1].set_xlabel("Time Step")
    axs[1].set_ylabel("L1 (cm)")
    fig.suptitle("Average PSNR: {:.2f}, Average Depth L1: {:.2f} cm, ATE RMSE: {:.2f} cm".format(avg_psnr, avg_l1*100, ate_rmse*100), y=1.05, fontsize=16)
    plt.savefig(os.path.join(eval_dir, "metrics.png"), bbox_inches='tight')
    if wandb_run is not None:
        wandb_run.log({"Eval/Metrics": fig})
    plt.close()

def eval_semantic_tree_newrender(dataset, final_params, num_frames, eval_dir, sil_thres, 
         mapping_iters, add_new_gaussians, wandb_run=None, 
         wandb_save_qual=False, eval_every=1, save_frames=False, flag_printtxt = True, 
         flag_mlp = 0, mlp_func = None, gt_transfer = False):
    print("Evaluating Final Parameters ...")
    print("using Renderer_semantic")
    psnr_list = []
    rmse_list = []
    l1_list = []
    lpips_list = []
    ssim_list = []
    miou_list = []
    mbiou_list = []
    plot_dir = os.path.join(eval_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    # if save_frames:
    render_rgb_dir = os.path.join(eval_dir, "rendered_rgb")
    os.makedirs(render_rgb_dir, exist_ok=True)
    render_depth_dir = os.path.join(eval_dir, "rendered_depth")
    os.makedirs(render_depth_dir, exist_ok=True)
    render_sem_dir = os.path.join(eval_dir, "rendered_semantic")
    os.makedirs(render_sem_dir, exist_ok=True)
    rgb_dir = os.path.join(eval_dir, "rgb")
    os.makedirs(rgb_dir, exist_ok=True)
    depth_dir = os.path.join(eval_dir, "depth")
    os.makedirs(depth_dir, exist_ok=True)

    gt_w2c_list = []
    for time_idx in tqdm(range(num_frames)):
        # Get RGB-D Data & Camera Parameters
        if dataset.use_pyramid:
            color, depth, intrinsics, pose, label_gt, pyramid_color, pyramid_depth, pyramid_semantic = dataset[time_idx]
        else:
            color, depth, intrinsics, pose, label_gt = dataset[time_idx]
        gt_w2c = torch.linalg.inv(pose)
        gt_w2c_list.append(gt_w2c)
        intrinsics = intrinsics[:3, :3]

        # Process RGB-D Data
        color = color.permute(2, 0, 1) / 255 # (H, W, C) -> (C, H, W)
        depth = depth.permute(2, 0, 1) # (H, W, C) -> (C, H, W)

        if time_idx == 0:
            # Process Camera Parameters
            first_frame_w2c = torch.linalg.inv(pose)
            # Setup Camera
            cam = setup_camera(color.shape[2], color.shape[1], intrinsics.cpu().numpy(), first_frame_w2c.detach().cpu().numpy())
        
        # Skip frames if not eval_every 
        if time_idx != 0 and (time_idx+1) % eval_every != 0:
            continue

        # Get current frame Gaussians
        transformed_pts = transform_to_frame(final_params, time_idx, 
                                             gaussians_grad=False,
                                             camera_grad=False)
 
        # Define current frame data
        curr_data = {'cam': cam, 'im': color, 'depth': depth, 'id': time_idx, 'intrinsics': intrinsics, 'w2c': first_frame_w2c, 'label_gt':label_gt}

        # Initialize Render Variables
        rendervar_sem = transformed_params2rendervar_semantic(final_params, transformed_pts)

        # 1. Render Depth & Silhouette
        im, radius, im_semantic, rendered_depth, rendered_median_depth, rendered_final_opcity =\
            Renderer_semantic(raster_settings=curr_data['cam'])(**rendervar_sem)

        if flag_mlp == 1:
            im_semantic_in = im_semantic.unsqueeze(0)                           # [1, num_emdedding, h, w]]
            logits = mlp_func(im_semantic_in).squeeze(0)                        # [num_classes, h, w]
            logits = F.softmax(logits, dim=0)
            im_semantic_label = torch.argmax(logits, dim=0)

        rastered_depth = rendered_depth
        # Mask invalid depth in GT
        valid_depth_mask = (curr_data['depth'] > 0)
        rastered_depth_viz = rastered_depth.detach()
        rastered_depth = rastered_depth * valid_depth_mask
        silhouette = rendered_final_opcity.squeeze(0)
        presence_sil_mask = (silhouette > sil_thres)
        
        if mapping_iters==0 and not add_new_gaussians:
            weighted_im = im * presence_sil_mask * valid_depth_mask
            weighted_gt_im = curr_data['im'] * presence_sil_mask * valid_depth_mask
        else:
            weighted_im = im * valid_depth_mask
            weighted_gt_im = curr_data['im'] * valid_depth_mask
        psnr = calc_psnr(weighted_im, weighted_gt_im).mean()
        ssim = ms_ssim(weighted_im.unsqueeze(0).cpu(), weighted_gt_im.unsqueeze(0).cpu(), 
                        data_range=1.0, size_average=True)
        lpips_score = loss_fn_alex(torch.clamp(weighted_im.unsqueeze(0), 0.0, 1.0),
                                    torch.clamp(weighted_gt_im.unsqueeze(0), 0.0, 1.0)).item()

        psnr_list.append(psnr.cpu().numpy())
        ssim_list.append(ssim.cpu().numpy())
        lpips_list.append(lpips_score)

        # Compute Depth RMSE
        if mapping_iters==0 and not add_new_gaussians:
            diff_depth_rmse = torch.sqrt((((rastered_depth - curr_data['depth']) * presence_sil_mask) ** 2))
            diff_depth_rmse = diff_depth_rmse * valid_depth_mask
            rmse = diff_depth_rmse.sum() / valid_depth_mask.sum()
            diff_depth_l1 = torch.abs((rastered_depth - curr_data['depth']) * presence_sil_mask)
            diff_depth_l1 = diff_depth_l1 * valid_depth_mask
            depth_l1 = diff_depth_l1.sum() / valid_depth_mask.sum()
        else:
            diff_depth_rmse = torch.sqrt((((rastered_depth - curr_data['depth'])) ** 2))
            diff_depth_rmse = diff_depth_rmse * valid_depth_mask
            rmse = diff_depth_rmse.sum() / valid_depth_mask.sum()
            diff_depth_l1 = torch.abs((rastered_depth - curr_data['depth']))
            diff_depth_l1 = diff_depth_l1 * valid_depth_mask
            depth_l1 = diff_depth_l1.sum() / valid_depth_mask.sum()
        rmse_list.append(rmse.cpu().numpy())
        l1_list.append(depth_l1.cpu().numpy())

        ## semantic loss
        logits_2_label = lambda x: torch.argmax(torch.nn.functional.softmax(x, dim=-1),dim=-1)
        
        start_time = time.time()
        iou_scores = {}  # Store IoU scores for each class
        biou_scores = {}
        class_counts = {}  # Count the number of times each class appears

        # semantic visualization (tree) --------------------
        if dataset.sem_mode == "nyu40":
            # origianl semantic visualization
            semantic_label = logits_2_label(im_semantic.permute(1,2,0))
            semantic_label_gt = curr_data['label_gt'].squeeze(-1)
        elif 'tree' in dataset.sem_mode and flag_mlp == 0:
            # flat
            im_semantic_treelabel = transfer_tree_label(im_semantic, dataset.num_semantic)  # [tree-flat, h, w] -> [num_level, h, w]
            if "replica" in dataset.name:
                im_semantic_label = transfer_tree_2_label(dataset, im_semantic_treelabel, flag_keystr=True) # [num_level, h, w] (tree) -> [h, w] (original)
            elif "scannet" in dataset.name:
                im_semantic_label = transfer_tree_2_label(dataset, im_semantic_treelabel)
            if save_frames:
                save_name = "sem_{:04d}_multilevel".format(time_idx)
                im_semantic_treelabel = im_semantic_treelabel.detach().cpu()
                if "replica" in dataset.name:
                    im_semantic_treelabel_vis = semantic_label_vis_replica(im_semantic_treelabel, dataset.colour_map_np_level)
                elif "scannet" in dataset.name:
                    im_semantic_treelabel_gt_vis = semantic_label_vis(im_semantic_treelabel_gt, dataset.colour_map_np_level)
                print("===> save at: ", os.path.join(render_sem_dir,  save_name+".png"))
                cv2.imwrite(os.path.join(render_sem_dir,  save_name+".png"), im_semantic_treelabel_vis[...,::-1])
            
            im_semantic_treelabel_gt = label_gt[:-1, :, :]
            semantic_label_gt = label_gt[-1, :, :]
            # semantic visualization (tree) --------------------         
        elif 'tree' in dataset.sem_mode and flag_mlp == 1:
            # replica - tree - mlp
            im_semantic_label_vis = im_semantic_label.detach().cpu().numpy()
            im_semantic_label_colorbar = dataset.colors_map_all[im_semantic_label_vis].astype(np.uint8) 
            im_semantic_label_colorbar = np.squeeze(im_semantic_label_colorbar) 
            if save_frames:
                save_name = "sem_{:04d}".format(time_idx)
                print("===> save at: ", os.path.join(render_sem_dir,  save_name+".png"))
                cv2.imwrite(os.path.join(render_sem_dir,  save_name+".png"), im_semantic_label_colorbar[...,::-1])
            
            im_semantic_treelabel_gt = label_gt[:-1, :, :]
            semantic_label_gt = label_gt[-1, :, :]
            
            im_semantic_treelabel_gt = im_semantic_treelabel_gt.detach().cpu()
            if "replica" in dataset.name:
                im_semantic_treelabel_gt_vis = semantic_label_vis_replica(im_semantic_treelabel_gt, dataset.colour_map_np_level)
            elif "scannet" in dataset.name:
                im_semantic_treelabel_gt_vis = semantic_label_vis(im_semantic_treelabel_gt, dataset.colour_map_np_level)
            if save_frames:
                save_name = "gt_{:04d}".format(time_idx)
                print("===> save at: ", os.path.join(render_sem_dir,  save_name+".png"))
                cv2.imwrite(os.path.join(render_sem_dir,  save_name+".png"), im_semantic_treelabel_gt_vis[...,::-1])

            # Use same eval proc as SGS-SLAM (True): ======
            # use per frame semantic gt to tranfer each frame's estimation
            # which will largely improve the miou score because there will be no iou=0 class situation.
            # Not Recommended.
            flag_color_transfer = gt_transfer
            if flag_color_transfer:
                # im_semantic_label_colorbar vs im_semantic_treelabel_gt_vis
                img_shape = im_semantic_treelabel_gt_vis.shape
                # gt_visualization
                im_semantic_treelabel_gt_vis_reshape = im_semantic_treelabel_gt_vis.reshape(-1, 3)
                color_map, _ = np.unique(im_semantic_treelabel_gt_vis_reshape, axis=0, return_inverse=True)
                refer_color = color_map.reshape(1, -1, 3).astype(np.float32)

                # estimation_visualization
                rendered_seg = im_semantic_label_colorbar.reshape(-1, 1, 3).astype(np.float32)

                # Find the index of the minimum distance for each pixel
                l1_distances = np.sqrt(np.sum((rendered_seg - refer_color) ** 2, axis=2))
                closest_indices = np.argmin(l1_distances, axis=1)
                del l1_distances

                # Assign the closest color to the rendered semantic image
                rendered_seg[:, 0, :] = refer_color.squeeze(0)[closest_indices]
                rendered_seg = rendered_seg.reshape(img_shape)  # (H*W, 1, 3) -> (H, W, 3)
                
                # save rendered_seg
                if save_frames:
                    save_name = "sem_{:04d}_color_transfer".format(time_idx)
                    print("===> save at: ", os.path.join(render_sem_dir,  save_name+".png"))
                    cv2.imwrite(os.path.join(render_sem_dir,  save_name+".png"), rendered_seg[...,::-1])

                # transfer visualization back to label_map
                color_to_label = {tuple(color): label for label, color in enumerate(dataset.colors_map_all)}
                H, W, _ = im_semantic_label_colorbar.shape
                im_semantic_label_transfer = np.zeros((H, W), dtype=np.int64)

                for i in range(H):
                    for j in range(W):
                        color_label = tuple(rendered_seg[i, j])
                        im_semantic_label_transfer[i, j] = color_to_label.get(color_label)
                im_semantic_label_transfer = torch.tensor(im_semantic_label_transfer, dtype=torch.int64).to(im_semantic_label.device)
                im_semantic_label = im_semantic_label_transfer


        # eval (1. using original classes to eval) ========
        flag_printsem = True
        if flag_printsem:
            print("current frame is: {}".format(time_idx))
        
        # use scannet-raw-id, which is sparse
        flag_raw_sem_mode = (dataset.sem_mode == 'tree_large')
        if "scannet" in dataset.dataset_name and flag_raw_sem_mode:
            for idx_sem_label, sem_label in enumerate(dataset.semantic_id):
                # transform label to binary mask
                sem_label_mask = (im_semantic_label == sem_label).float()
                sem_label_gt_mask = (semantic_label_gt == sem_label).float()
                if sem_label_mask.sum() == 0 and sem_label_gt_mask.sum() == 0:
                    continue
                
                sem_label_mask = sem_label_mask.cpu().numpy()
                sem_label_gt_mask = sem_label_gt_mask.cpu().numpy()
                
                # compute iou
                iou = calculate_iou(sem_label_gt_mask, sem_label_mask)
                biou = boundary_iou(sem_label_gt_mask, sem_label_mask)
                # accumulate iou
                if idx_sem_label not in iou_scores:
                    iou_scores[idx_sem_label] = []
                    biou_scores[idx_sem_label] = []
                iou_scores[idx_sem_label].append(iou)
                biou_scores[idx_sem_label].append(biou)
                class_counts[idx_sem_label] = class_counts.get(sem_label, 0) + 1
                if flag_printsem:
                    if iou==0:
                        if "scannet" in dataset.dataset_name and dataset.sem_mode in 'tree':
                            print(" semantic label {} ({}): iou: {:.3f}, biou: {:.3f}, class_counts: {}, pixel num gt vs est: {} vs {}".format(sem_label, class_name_string_nyu40[sem_label], iou, biou, class_counts[sem_label], sem_label_gt_mask.sum(), sem_label_mask.sum()))
                        elif "scannet" in dataset.dataset_name and dataset.sem_mode in 'tree_large':
                            print(" semantic label {} ({}): iou: {:.3f}, biou: {:.3f}, class_counts: {}, pixel num gt vs est: {} vs {}".format(sem_label, dataset.semantic_class[idx_sem_label], iou, biou, class_counts[idx_sem_label], sem_label_gt_mask.sum(), sem_label_mask.sum()))
                        elif "replica" in dataset.dataset_name:
                            print(" semantic label {} ({}): iou: {:.3f}, biou: {:.3f}, class_counts: {}, pixel num gt vs est: {} vs {}".format(sem_label, dataset.tree_id_classes_map[-1][sem_label], iou, biou, class_counts[sem_label], sem_label_gt_mask.sum(), sem_label_mask.sum()))
                    
                    else:
                        if "scannet" in dataset.dataset_name:
                            if dataset.sem_mode == 'tree_large':
                                print(" semantic label {} ({}): iou: {:.3f}, biou: {:.3f}, class_counts: {}".format(sem_label, dataset.semantic_class[idx_sem_label], iou, biou, class_counts[idx_sem_label]))
                            elif dataset.sem_mode == 'tree':
                                print(" semantic label {} ({}): iou: {:.3f}, biou: {:.3f}, class_counts: {}".format(sem_label, class_name_string_nyu40[sem_label], iou, biou, class_counts[sem_label]))
                        elif "replica" in dataset.dataset_name:
                            print(" semantic label {} ({}): iou: {:.3f}, biou: {:.3f}, class_counts: {}".format(sem_label, dataset.tree_id_classes_map[-1][sem_label], iou, biou, class_counts[sem_label]))

        else:
            # replica
            for sem_label in range(dataset.num_semantic_class):
                # transform label to binary mask
                sem_label_mask = (im_semantic_label == sem_label).float()
                sem_label_gt_mask = (semantic_label_gt == sem_label).float()
                if sem_label_mask.sum() == 0 and sem_label_gt_mask.sum() == 0:
                    continue
                
                sem_label_mask = sem_label_mask.cpu().numpy()
                sem_label_gt_mask = sem_label_gt_mask.cpu().numpy()
                visualize_label(sem_label_mask, dataset.colors_map_all, flag_save = False, save_name = os.path.join(render_sem_dir, "sem_{:04d}_class{}".format(time_idx, sem_label)) )
                visualize_label(sem_label_gt_mask, dataset.colors_map_all, flag_save = False, save_name = os.path.join(render_sem_dir, "gt_{:04d}_class{}".format(time_idx, sem_label)) )
                
                # compute iou
                iou = calculate_iou(sem_label_gt_mask, sem_label_mask)
                biou = boundary_iou(sem_label_gt_mask, sem_label_mask)
                # accumulate iou
                if sem_label not in iou_scores:
                    iou_scores[sem_label] = []
                    biou_scores[sem_label] = []
                iou_scores[sem_label].append(iou)
                biou_scores[sem_label].append(biou)
                class_counts[sem_label] = class_counts.get(sem_label, 0) + 1
                if flag_printsem:
                    if iou==0:
                        if "scannet" in dataset.dataset_name and dataset.sem_mode == 'tree':
                            print(" semantic label {} ({}): iou: {:.3f}, biou: {:.3f}, class_counts: {}, pixel num gt vs est: {} vs {}".format(sem_label, class_name_string_nyu40[sem_label], iou, biou, class_counts[sem_label], sem_label_gt_mask.sum(), sem_label_mask.sum()))
                        elif "scannet" in dataset.dataset_name and dataset.sem_mode == 'tree_large':
                            print(" semantic label {} ({}): iou: {:.3f}, biou: {:.3f}, class_counts: {}, pixel num gt vs est: {} vs {}".format(sem_label, dataset.semantic_class[sem_label], iou, biou, class_counts[sem_label], sem_label_gt_mask.sum(), sem_label_mask.sum()))
                        elif "replica" in dataset.dataset_name:
                            print(" semantic label {} ({}): iou: {:.3f}, biou: {:.3f}, class_counts: {}, pixel num gt vs est: {} vs {}".format(sem_label, dataset.tree_id_classes_map[-1][sem_label], iou, biou, class_counts[sem_label], sem_label_gt_mask.sum(), sem_label_mask.sum()))
                    
                    else:
                        if "scannet" in dataset.dataset_name:
                            if dataset.sem_mode == 'tree_large':
                                print(" semantic label {} ({}): iou: {:.3f}, biou: {:.3f}, class_counts: {}".format(sem_label, dataset.semantic_class[sem_label], iou, biou, class_counts[sem_label]))
                            elif dataset.sem_mode == 'tree':
                                print(" semantic label {} ({}): iou: {:.3f}, biou: {:.3f}, class_counts: {}".format(sem_label, class_name_string_nyu40[sem_label], iou, biou, class_counts[sem_label]))
                        elif "replica" in dataset.dataset_name:
                            print(" semantic label {} ({}): iou: {:.3f}, biou: {:.3f}, class_counts: {}".format(sem_label, dataset.tree_id_classes_map[-1][sem_label], iou, biou, class_counts[sem_label]))
            
        # Calculate mean IoU for each class
        mean_iou_per_class = {cat_id: np.mean(iou_scores[cat_id]) for cat_id in iou_scores}
        mean_biou_per_class = {cat_id: np.mean(biou_scores[cat_id]) for cat_id in biou_scores}

        # Calculate overall mean IoU
        overall_mean_iou = np.mean(list(mean_iou_per_class.values()))
        overall_mean_biou = np.mean(list(mean_biou_per_class.values()))

        if flag_printsem:
            print("mean_iou: {:.4f}, mean_biou: {:.4f}".format(overall_mean_iou, overall_mean_biou))

        miou_list.append(overall_mean_iou)
        mbiou_list.append(overall_mean_biou)

        # save => txt
        if flag_printtxt:
            with open(os.path.join(render_sem_dir, "sem_iou_2flat.txt"), 'a') as f:
                f.write("frame: {}\n".format(time_idx))
                f.write("mean_iou: {:.4f}, mean_biou: {:.4f}\n".format(overall_mean_iou, overall_mean_biou))
                f.write("mean_iou_per_class: {}\n".format(mean_iou_per_class))
                f.write("mean_biou_per_class: {}\n".format(mean_biou_per_class))
                f.write("\n")
                pass

        if False:
            print("Mean IoU per class:", mean_iou_per_class)
            print("Mean Boundary IoU per class:", mean_biou_per_class)
            print("Overall Mean IoU:", overall_mean_iou)
            print("Overall Boundary Mean IoU:", overall_mean_biou)
        # =================================================

        if save_frames:
            # Save Rendered RGB and Depth
            viz_render_im = torch.clamp(im, 0, 1)
            viz_render_im = viz_render_im.detach().cpu().permute(1, 2, 0).numpy()
            vmin = 0
            vmax = 6
            viz_render_depth = rastered_depth_viz[0].detach().cpu().numpy()
            normalized_depth = np.clip((viz_render_depth - vmin) / (vmax - vmin), 0, 1)
            depth_colormap = cv2.applyColorMap((normalized_depth * 255).astype(np.uint8), cv2.COLORMAP_JET)
            cv2.imwrite(os.path.join(render_rgb_dir, "gs_{:04d}.png".format(time_idx)), cv2.cvtColor(viz_render_im*255, cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(render_depth_dir, "gs_{:04d}.png".format(time_idx)), depth_colormap)

            # Save GT RGB and Depth
            viz_gt_im = torch.clamp(curr_data['im'], 0, 1)
            viz_gt_im = viz_gt_im.detach().cpu().permute(1, 2, 0).numpy()
            viz_gt_depth = curr_data['depth'][0].detach().cpu().numpy()
            normalized_depth = np.clip((viz_gt_depth - vmin) / (vmax - vmin), 0, 1)
            depth_colormap = cv2.applyColorMap((normalized_depth * 255).astype(np.uint8), cv2.COLORMAP_JET)
            cv2.imwrite(os.path.join(rgb_dir, "gt_{:04d}.png".format(time_idx)), cv2.cvtColor(viz_gt_im*255, cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(depth_dir, "gt_{:04d}.png".format(time_idx)), depth_colormap)
        
        # Plot the Ground Truth and Rasterized RGB & Depth, along with Silhouette
        fig_title = "Time Step: {}".format(time_idx)
        plot_name = "%04d" % time_idx
        presence_sil_mask = presence_sil_mask.detach().cpu().numpy()
        if wandb_run is None:
            plot_rgbd_silhouette(color, depth, im, rastered_depth_viz, presence_sil_mask, diff_depth_l1,
                                 psnr, depth_l1, fig_title, plot_dir, 
                                 plot_name=plot_name, save_plot=True)
        elif wandb_save_qual:
            plot_rgbd_silhouette(color, depth, im, rastered_depth_viz, presence_sil_mask, diff_depth_l1,
                                 psnr, depth_l1, fig_title, plot_dir, 
                                 plot_name=plot_name, save_plot=True,
                                 wandb_run=wandb_run, wandb_step=None, 
                                 wandb_title="Eval/Qual Viz")

    try:
        # Compute the final ATE RMSE
        # Get the final camera trajectory
        num_frames = final_params['cam_unnorm_rots'].shape[-1]
        latest_est_w2c = first_frame_w2c
        latest_est_w2c_list = []
        latest_est_w2c_list.append(latest_est_w2c)
        valid_gt_w2c_list = []
        valid_gt_w2c_list.append(gt_w2c_list[0])
        for idx in range(1, num_frames):
            # Check if gt pose is not nan for this time step
            if torch.isnan(gt_w2c_list[idx]).sum() > 0:
                continue
            interm_cam_rot = F.normalize(final_params['cam_unnorm_rots'][..., idx].detach())
            interm_cam_trans = final_params['cam_trans'][..., idx].detach()
            intermrel_w2c = torch.eye(4).cuda().float()
            intermrel_w2c[:3, :3] = build_rotation(interm_cam_rot)
            intermrel_w2c[:3, 3] = interm_cam_trans
            latest_est_w2c = intermrel_w2c
            latest_est_w2c_list.append(latest_est_w2c)
            valid_gt_w2c_list.append(gt_w2c_list[idx])
        gt_w2c_list = valid_gt_w2c_list
        # Calculate ATE RMSE
        ate_rmse = evaluate_ate(gt_w2c_list, latest_est_w2c_list)
        print("Final Average ATE RMSE: {:.2f} cm".format(ate_rmse*100))
        if wandb_run is not None:
            wandb_run.log({"Final Stats/Avg ATE RMSE": ate_rmse,
                        "Final Stats/step": 1})
    except:
        ate_rmse = 100.0
        print('Failed to evaluate trajectory with alignment.')
    
    # Compute Average Metrics
    psnr_list = np.array(psnr_list)
    rmse_list = np.array(rmse_list)
    l1_list = np.array(l1_list)
    ssim_list = np.array(ssim_list)
    lpips_list = np.array(lpips_list)
    miou_list = np.array(miou_list)
    mbiou_list = np.array(mbiou_list)
    avg_psnr = psnr_list.mean()
    avg_rmse = rmse_list.mean()
    avg_l1 = l1_list.mean()
    avg_ssim = ssim_list.mean()
    avg_lpips = lpips_list.mean()
    avg_miou = miou_list.mean()         # for each frame, paper use every 4 frames/randomly 100 frames. here every 500 frames
    avg_mbiou = mbiou_list.mean()
    print("Average PSNR: {:.2f}".format(avg_psnr))
    print("Average MS-SSIM: {:.3f}".format(avg_ssim))
    print("Average LPIPS: {:.3f}".format(avg_lpips))
    print("Average Depth L1: {:.2f} cm".format(avg_l1*100))
    print("Average Depth RMSE: {:.2f} cm".format(avg_rmse*100))
    print("Average miou: {:.3f}".format(avg_miou*100.0))
    print("Average mbiou: {:.3f}".format(avg_mbiou*100.0))
    print(" ==== summary ==== ")
    print("[ATE RMSE] [PSNR] [MS-SSIM] [LPIPS] [Depth L1] [Depth RMSE] [miou%] [mbiou%]")
    print("{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}".format(
        ate_rmse*100, avg_psnr, avg_ssim, avg_lpips, avg_l1*100, avg_rmse*100, avg_miou*100.0, avg_mbiou*100.0))

    if wandb_run is not None:
        wandb_run.log({"Final Stats/Average PSNR": avg_psnr, 
                       "Final Stats/Average MS-SSIM": avg_ssim, 
                       "Final Stats/Average LPIPS": avg_lpips,
                       "Final Stats/Average Depth L1": avg_l1,
                       "Final Stats/Average Depth RMSE": avg_rmse,
                       "Final Stats/semantic miou": avg_miou*100.0,
                       "Final Stats/semantic mbiou": avg_mbiou*100.0,
                       "Final Stats/step": 1})

    # Save metric lists as text files
    np.savetxt(os.path.join(eval_dir, "psnr.txt"), psnr_list)
    np.savetxt(os.path.join(eval_dir, "rmse.txt"), rmse_list)
    np.savetxt(os.path.join(eval_dir, "l1.txt"), l1_list)
    np.savetxt(os.path.join(eval_dir, "ssim.txt"), ssim_list)
    np.savetxt(os.path.join(eval_dir, "lpips.txt"), lpips_list)
    np.savetxt(os.path.join(eval_dir, "miou.txt"), miou_list)       
    np.savetxt(os.path.join(eval_dir, "mbiou.txt"), mbiou_list)     

    # Plot PSNR & L1 as line plots
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    axs[0].plot(np.arange(len(psnr_list)), psnr_list)
    axs[0].set_title("RGB PSNR")
    axs[0].set_xlabel("Time Step")
    axs[0].set_ylabel("PSNR")
    axs[1].plot(np.arange(len(l1_list)), l1_list*100)
    axs[1].set_title("Depth L1")
    axs[1].set_xlabel("Time Step")
    axs[1].set_ylabel("L1 (cm)")
    fig.suptitle("Average PSNR: {:.2f}, Average Depth L1: {:.2f} cm, ATE RMSE: {:.2f} cm".format(avg_psnr, avg_l1*100, ate_rmse*100), y=1.05, fontsize=16)
    plt.savefig(os.path.join(eval_dir, "metrics.png"), bbox_inches='tight')
    if wandb_run is not None:
        wandb_run.log({"Eval/Metrics": fig})
    plt.close()

def eval_nvs(dataset, final_params, num_frames, eval_dir, sil_thres, 
         mapping_iters, add_new_gaussians, wandb_run=None, wandb_save_qual=False, eval_every=1, save_frames=False):
    print("Evaluating Final Parameters for Novel View Synthesis ...")
    psnr_list = []
    rmse_list = []
    l1_list = []
    lpips_list = []
    ssim_list = []
    valid_nvs_frames = []
    plot_dir = os.path.join(eval_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    if save_frames:
        render_rgb_dir = os.path.join(eval_dir, "rendered_rgb")
        os.makedirs(render_rgb_dir, exist_ok=True)
        render_depth_dir = os.path.join(eval_dir, "rendered_depth")
        os.makedirs(render_depth_dir, exist_ok=True)
        rgb_dir = os.path.join(eval_dir, "rgb")
        os.makedirs(rgb_dir, exist_ok=True)
        depth_dir = os.path.join(eval_dir, "depth")
        os.makedirs(depth_dir, exist_ok=True)

    for time_idx in tqdm(range(num_frames)):
         # Get RGB-D Data & Camera Parameters
        color, depth, intrinsics, pose = dataset[time_idx]
        gt_w2c = torch.linalg.inv(pose)
        intrinsics = intrinsics[:3, :3]

        # Process RGB-D Data
        color = color.permute(2, 0, 1) / 255 # (H, W, C) -> (C, H, W)
        depth = depth.permute(2, 0, 1) # (H, W, C) -> (C, H, W)

        if time_idx == 0:
            # Process Camera Parameters
            first_frame_w2c = torch.linalg.inv(pose)
            # Setup Camera
            cam = setup_camera(color.shape[2], color.shape[1], intrinsics.cpu().numpy(), first_frame_w2c.detach().cpu().numpy())
            # Skip first train frame eval for NVS
            continue
        
        # Skip frames if not eval_every (indexing accounts for first training frame)
        test_time_idx = time_idx - 1
        if test_time_idx != 0 and (test_time_idx+1) % eval_every != 0:
            continue

        transformed_gaussians = {}
        # Transform Centers of Gaussians to Camera Frame
        pts = final_params['means3D'].detach()
        pts_ones = torch.ones(pts.shape[0], 1).cuda().float()
        pts4 = torch.cat((pts, pts_ones), dim=1)
        transformed_pts = (gt_w2c @ pts4.T).T[:, :3]
        transformed_gaussians['means3D'] = transformed_pts
        # Check if Gaussians need to be rotated (Isotropic or Anisotropic)
        if final_params['log_scales'].shape[1] == 1:
            transform_rots = False # Isotropic Gaussians
        else:
            transform_rots = True # Anisotropic Gaussians
        # Transform Rots of Gaussians to Camera Frame
        if transform_rots:
            norm_rots = F.normalize(final_params['unnorm_rotations'].detach())
            gt_cam_rot = matrix_to_quaternion(gt_w2c[:3, :3])
            gt_cam_rot = F.normalize(gt_cam_rot.unsqueeze(0))
            transformed_rots = quat_mult(gt_cam_rot, norm_rots)
            transformed_gaussians['unnorm_rotations'] = transformed_rots
        else:
            transformed_gaussians['unnorm_rotations'] = final_params['unnorm_rotations'].detach()
 
        # Define current frame data
        curr_data = {'cam': cam, 'im': color, 'depth': depth, 'id': time_idx, 'intrinsics': intrinsics, 'w2c': first_frame_w2c}

        # Initialize Render Variables
        rendervar = transformed_params2rendervar(final_params, transformed_gaussians)
        depth_sil_rendervar = transformed_params2depthplussilhouette(final_params, curr_data['w2c'],
                                                                     transformed_gaussians)

        # Render Depth & Silhouette
        depth_sil, _, _, = Renderer(raster_settings=curr_data['cam'])(**depth_sil_rendervar)
        rastered_depth = depth_sil[0, :, :].unsqueeze(0)
        # Mask invalid depth in GT
        valid_depth_mask = (curr_data['depth'] > 0)
        rastered_depth_viz = rastered_depth.detach()
        rastered_depth = rastered_depth * valid_depth_mask
        silhouette = depth_sil[1, :, :]
        presence_sil_mask = (silhouette > sil_thres)

        # Check if Novel View is Valid based on Silhouette & Valid Depth Mask
        valid_region_mask = presence_sil_mask | ~valid_depth_mask
        percent_holes = (~valid_region_mask).sum() / valid_region_mask.numel() * 100
        if percent_holes > 0.1:
            valid_nvs_frames.append(False)
        else:
            valid_nvs_frames.append(True)
        
        # Render RGB and Calculate PSNR
        im, radius, _, = Renderer(raster_settings=curr_data['cam'])(**rendervar)
        if mapping_iters==0 and not add_new_gaussians:
            weighted_im = im * presence_sil_mask * valid_depth_mask
            weighted_gt_im = curr_data['im'] * presence_sil_mask * valid_depth_mask
        else:
            weighted_im = im * valid_depth_mask
            weighted_gt_im = curr_data['im'] * valid_depth_mask
        diff_rgb = torch.abs(weighted_im - weighted_gt_im).mean(dim=0).detach()
        psnr = calc_psnr(weighted_im, weighted_gt_im).mean()
        ssim = ms_ssim(weighted_im.unsqueeze(0).cpu(), weighted_gt_im.unsqueeze(0).cpu(), 
                        data_range=1.0, size_average=True)
        lpips_score = loss_fn_alex(torch.clamp(weighted_im.unsqueeze(0), 0.0, 1.0),
                                    torch.clamp(weighted_gt_im.unsqueeze(0), 0.0, 1.0)).item()

        psnr_list.append(psnr.cpu().numpy())
        ssim_list.append(ssim.cpu().numpy())
        lpips_list.append(lpips_score)

        # Compute Depth RMSE
        if mapping_iters==0 and not add_new_gaussians:
            diff_depth_rmse = torch.sqrt((((rastered_depth - curr_data['depth']) * presence_sil_mask) ** 2))
            diff_depth_rmse = diff_depth_rmse * valid_depth_mask
            rmse = diff_depth_rmse.sum() / valid_depth_mask.sum()
            diff_depth_l1 = torch.abs((rastered_depth - curr_data['depth']) * presence_sil_mask)
            diff_depth_l1 = diff_depth_l1 * valid_depth_mask
            depth_l1 = diff_depth_l1.sum() / valid_depth_mask.sum()
        else:
            diff_depth_rmse = torch.sqrt((((rastered_depth - curr_data['depth'])) ** 2))
            diff_depth_rmse = diff_depth_rmse * valid_depth_mask
            rmse = diff_depth_rmse.sum() / valid_depth_mask.sum()
            diff_depth_l1 = torch.abs((rastered_depth - curr_data['depth']))
            diff_depth_l1 = diff_depth_l1 * valid_depth_mask
            depth_l1 = diff_depth_l1.sum() / valid_depth_mask.sum()
        rmse_list.append(rmse.cpu().numpy())
        l1_list.append(depth_l1.cpu().numpy())

        if save_frames:
            # Save Rendered RGB and Depth
            viz_render_im = torch.clamp(im, 0, 1)
            viz_render_im = viz_render_im.detach().cpu().permute(1, 2, 0).numpy()
            vmin = 0
            vmax = 6
            viz_render_depth = rastered_depth_viz[0].detach().cpu().numpy()
            normalized_depth = np.clip((viz_render_depth - vmin) / (vmax - vmin), 0, 1)
            depth_colormap = cv2.applyColorMap((normalized_depth * 255).astype(np.uint8), cv2.COLORMAP_JET)
            cv2.imwrite(os.path.join(render_rgb_dir, "render_{:04d}.png".format(test_time_idx)), cv2.cvtColor(viz_render_im*255, cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(render_depth_dir, "render_{:04d}.png".format(test_time_idx)), depth_colormap)

            # Save GT RGB and Depth
            viz_gt_im = torch.clamp(curr_data['im'], 0, 1)
            viz_gt_im = viz_gt_im.detach().cpu().permute(1, 2, 0).numpy()
            viz_gt_depth = curr_data['depth'][0].detach().cpu().numpy()
            normalized_depth = np.clip((viz_gt_depth - vmin) / (vmax - vmin), 0, 1)
            depth_colormap = cv2.applyColorMap((normalized_depth * 255).astype(np.uint8), cv2.COLORMAP_JET)
            cv2.imwrite(os.path.join(rgb_dir, "gt_{:04d}.png".format(test_time_idx)), cv2.cvtColor(viz_gt_im*255, cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(depth_dir, "gt_{:04d}.png".format(test_time_idx)), depth_colormap)
        
        # Plot the Ground Truth and Rasterized RGB & Depth, along with Silhouette
        fig_title = "Time Step: {}".format(test_time_idx)
        plot_name = "%04d" % test_time_idx
        presence_sil_mask = presence_sil_mask.detach().cpu().numpy()
        if wandb_run is None:
            plot_rgbd_silhouette(color, depth, im, rastered_depth_viz, presence_sil_mask, diff_depth_l1,
                                 psnr, depth_l1, fig_title, plot_dir, 
                                 plot_name=plot_name, save_plot=True)
        elif wandb_save_qual:
            plot_rgbd_silhouette(color, depth, im, rastered_depth_viz, presence_sil_mask, diff_depth_l1,
                                 psnr, depth_l1, fig_title, plot_dir, 
                                 plot_name=plot_name, save_plot=True,
                                 wandb_run=wandb_run, wandb_step=None, 
                                 wandb_title="Eval/Qual Viz")

    # Compute Average Metrics based on valid NVS frames
    psnr_list = np.array(psnr_list)
    rmse_list = np.array(rmse_list)
    l1_list = np.array(l1_list)
    ssim_list = np.array(ssim_list)
    lpips_list = np.array(lpips_list)
    valid_nvs_frames = np.array(valid_nvs_frames)
    avg_psnr = psnr_list[valid_nvs_frames].mean()
    avg_rmse = rmse_list[valid_nvs_frames].mean()
    avg_l1 = l1_list[valid_nvs_frames].mean()
    avg_ssim = ssim_list[valid_nvs_frames].mean()
    avg_lpips = lpips_list[valid_nvs_frames].mean()
    print("Average PSNR: {:.2f}".format(avg_psnr))
    print("Average Depth RMSE: {:.2f} cm".format(avg_rmse*100))
    print("Average Depth L1: {:.2f} cm".format(avg_l1*100))
    print("Average MS-SSIM: {:.3f}".format(avg_ssim))
    print("Average LPIPS: {:.3f}".format(avg_lpips))

    if wandb_run is not None:
        wandb_run.log({"Final Stats/Average PSNR": avg_psnr, 
                       "Final Stats/Average Depth RMSE": avg_rmse,
                       "Final Stats/Average Depth L1": avg_l1,
                       "Final Stats/Average MS-SSIM": avg_ssim, 
                       "Final Stats/Average LPIPS": avg_lpips,
                       "Final Stats/step": 1})

    # Save metric lists as text files
    np.savetxt(os.path.join(eval_dir, "psnr.txt"), psnr_list)
    np.savetxt(os.path.join(eval_dir, "rmse.txt"), rmse_list)
    np.savetxt(os.path.join(eval_dir, "l1.txt"), l1_list)
    np.savetxt(os.path.join(eval_dir, "ssim.txt"), ssim_list)
    np.savetxt(os.path.join(eval_dir, "lpips.txt"), lpips_list)

    # Save metadata for valid NVS frames
    np.save(os.path.join(eval_dir, "valid_nvs_frames.npy"), valid_nvs_frames)

    # Plot PSNR & L1 as line plots
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    axs[0].plot(np.arange(len(psnr_list)), psnr_list)
    axs[0].set_title("RGB PSNR")
    axs[0].set_xlabel("Time Step")
    axs[0].set_ylabel("PSNR")
    axs[1].plot(np.arange(len(l1_list)), l1_list*100)
    axs[1].set_title("Depth L1")
    axs[1].set_xlabel("Time Step")
    axs[1].set_ylabel("L1 (cm)")
    fig.suptitle("Average PSNR: {:.2f}, Average Depth L1: {:.2f} cm".format(avg_psnr, avg_l1*100), y=1.05, fontsize=16)
    plt.savefig(os.path.join(eval_dir, "metrics.png"), bbox_inches='tight')
    if wandb_run is not None:
        wandb_run.log({"Eval/Metrics": fig})
    plt.close()

def eval_semantic_single(dataset, im_semantic_label, im_semantic_label_gt):
    """
    im_semantic_label:    [h,w]
    im_semantic_label_gt: [h,w]
    """

    iou_scores = {}  # Store IoU scores for each class
    biou_scores = {}
    class_counts = {}  # Count the number of times each class appears
    flag_printsem = True
    num_wrong = 0
    render_sem_dir = "./experiments/Replica_semantic/test"

    for sem_label in range(dataset.num_semantic_class):
        # transform label to binary mask
        sem_label_mask = (im_semantic_label == sem_label).float()
        sem_label_gt_mask = (im_semantic_label_gt == sem_label).float()
        if sem_label_mask.sum() == 0 and sem_label_gt_mask.sum() == 0:
            continue
        
        sem_label_mask = sem_label_mask.cpu().numpy()
        sem_label_gt_mask = sem_label_gt_mask.cpu().numpy()
        
        if sem_label_mask.sum() == 0 or sem_label_gt_mask.sum() == 0:
            visualize_label(sem_label_mask, dataset.colors_map_all, flag_save = False, save_name = os.path.join(render_sem_dir, "sem_class{}".format(sem_label)) )
            visualize_label(sem_label_gt_mask, dataset.colors_map_all, flag_save = False, save_name = os.path.join(render_sem_dir, "gt_class{}".format(sem_label)) )
            pass

        # compute iou
        iou = calculate_iou(sem_label_gt_mask, sem_label_mask)
        biou = boundary_iou(sem_label_gt_mask, sem_label_mask)
        # accumulate iou
        if sem_label not in iou_scores:
            iou_scores[sem_label] = []
            biou_scores[sem_label] = []
        iou_scores[sem_label].append(iou)
        biou_scores[sem_label].append(biou)
        class_counts[sem_label] = class_counts.get(sem_label, 0) + 1
        if iou==0:
            num_wrong += sem_label_mask.sum()
        if flag_printsem:
            if iou==0:
                if "scannet" in dataset.dataset_name:
                    print(" semantic label {} ({}): iou: {:.3f}, biou: {:.3f}, class_counts: {}, pixel num gt vs est: {} vs {}".format(sem_label, dataset.semantic_class[sem_label], iou, biou, class_counts[sem_label], sem_label_gt_mask.sum(), sem_label_mask.sum()))
                elif "replica" in dataset.dataset_name:
                    print(" semantic label {} ({}): iou: {:.3f}, biou: {:.3f}, class_counts: {}, pixel num gt vs est: {} vs {}".format(sem_label, dataset.tree_id_classes_map[-1][sem_label], iou, biou, class_counts[sem_label], sem_label_gt_mask.sum(), sem_label_mask.sum()))
                    
            else:
                if "scannet" in dataset.dataset_name:
                    print(" semantic label {} ({}): iou: {:.3f}, biou: {:.3f}, class_counts: {}".format(sem_label, class_name_string_nyu40[sem_label], iou, biou, class_counts[sem_label]))
                elif "replica" in dataset.dataset_name:
                    if dataset.sem_mode == "tree":
                        print(" semantic label {} ({}): iou: {:.3f}, biou: {:.3f}, class_counts: {}".format(sem_label, dataset.tree_id_classes_map[-1][sem_label], iou, biou, class_counts[sem_label]))
                    elif dataset.sem_mode == "original":
                        print(" semantic label {} ({}): iou: {:.3f}, biou: {:.3f}, class_counts: {}".format(sem_label, dataset.semantic_class[sem_label], iou, biou, class_counts[sem_label]))

    # Calculate mean IoU for each class
    mean_iou_per_class = {cat_id: np.mean(iou_scores[cat_id]) for cat_id in iou_scores}
    mean_biou_per_class = {cat_id: np.mean(biou_scores[cat_id]) for cat_id in biou_scores}

    # Calculate overall mean IoU
    overall_mean_iou = np.mean(list(mean_iou_per_class.values()))
    overall_mean_biou = np.mean(list(mean_biou_per_class.values()))

    if True:
        print("mean_iou: {:.4f}, mean_biou: {:.4f}".format(overall_mean_iou, overall_mean_biou))
        print("num 0 worng is: ", num_wrong)

def show_semantic(dataset, final_params, num_frames, eval_dir, 
                flag_mlp = 0, mlp_func = None):
    
    flag_show_multilevel = True
    flag_use_method = 1 # 0: use method original, 1: use mlp trnasfer back
    flag_show_transparent = 1   # 0: no rgb-base; 1: rgb-base
    flag_onlyuse = True
    w_color = 0.35
    w_sem = 0.65
    # room0
    name_show = [4, 29]
    
    plot_dir = os.path.join(eval_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    if flag_use_method==0:
        render_sem_dir = os.path.join(eval_dir, "rendered_semantic_multilevel")
    if flag_use_method==1:
        render_sem_dir = os.path.join(eval_dir, "rendered_semantic_multilevel_mlp")
    os.makedirs(render_sem_dir, exist_ok=True)

    gt_w2c_list = []
    for time_idx in tqdm(range(num_frames)):
        if flag_onlyuse and not time_idx==0:
            if time_idx not in name_show:
                continue

        # Get RGB-D Data & Camera Parameters
        if dataset.use_pyramid:
            color, depth, intrinsics, pose, label_gt, pyramid_color, pyramid_depth, pyramid_semantic = dataset[time_idx]
        else:
            color, depth, intrinsics, pose, label_gt = dataset[time_idx]
        gt_w2c = torch.linalg.inv(pose)
        gt_w2c_list.append(gt_w2c)
        intrinsics = intrinsics[:3, :3]

        # Process RGB-D Data
        color = color.permute(2, 0, 1) / 255 # (H, W, C) -> (C, H, W)
        depth = depth.permute(2, 0, 1) # (H, W, C) -> (C, H, W)

        if time_idx == 0:
            # Process Camera Parameters
            first_frame_w2c = torch.linalg.inv(pose)
            # Setup Camera
            cam = setup_camera(color.shape[2], color.shape[1], intrinsics.cpu().numpy(), first_frame_w2c.detach().cpu().numpy())

        # Get current frame Gaussians
        transformed_pts = transform_to_frame(final_params, time_idx, 
                                             gaussians_grad=False,
                                             camera_grad=False)
        
        # Define current frame data
        curr_data = {'cam': cam, 'im': color, 'depth': depth, 'id': time_idx, 'intrinsics': intrinsics, 'w2c': first_frame_w2c, 'label_gt':label_gt}

        # Initialize Render Variables
        rendervar_sem = transformed_params2rendervar_semantic(final_params, transformed_pts)

        # 1. Render Depth & Silhouette
        im, radius, im_semantic, rendered_depth, rendered_median_depth, rendered_final_opcity =\
        Renderer_semantic(raster_settings=curr_data['cam'])(**rendervar_sem)

        if flag_mlp == 1:
            im_semantic_in = im_semantic.unsqueeze(0)                           # [1, num_emdedding, h, w]]
            logits = mlp_func(im_semantic_in).squeeze(0)                        # [num_classes, h, w]
            logits = F.softmax(logits, dim=0)
            im_semantic_label_mlp = torch.argmax(logits, dim=0)
            im_semantic_label_mlp = im_semantic_label_mlp.detach().cpu()

        # each level ========================
        im_semantic_treelabel = transfer_tree_label(im_semantic, dataset.num_semantic)
        im_semantic_treelabel = im_semantic_treelabel.detach().cpu()
        im_semantic_gt = curr_data['label_gt'].detach().cpu()

        if flag_show_multilevel and flag_use_method==0:
            # use original multi-level method
            save_name_base = "sem_{:04d}".format(time_idx)
            
            # level0 -- (level-end-1)
            for i_level in range(len(dataset.num_semantic)-2):
                # 1. data
                im_semantic_level, num_dim_est = transfer_eachlevel_1(im_semantic_treelabel[:i_level+1, :, :], dataset)
                im_semantic_gt_level, num_dim_gt = transfer_eachlevel_1(im_semantic_gt[:i_level+1, :, :], dataset)
                num_dim = max(num_dim_est, num_dim_gt)
                # 2. color_map
                label_color_map = label_colormap(num_dim)
                label_map_vis = label_color_map[im_semantic_level]
                label_map_gt_vis = label_color_map[im_semantic_gt_level]
                # save
                save_name = save_name_base+"_level{}".format(i_level)
                print("===> save at: ", os.path.join(render_sem_dir,  save_name+".png"))
                print("===> save at: ", os.path.join(render_sem_dir,  save_name+"_gt.png"))
                cv2.imwrite(os.path.join(render_sem_dir,  save_name+".png"), label_map_vis[...,::-1])
                cv2.imwrite(os.path.join(render_sem_dir,  save_name+"_gt.png"), label_map_gt_vis[...,::-1])
                
            # level-end
            im_semantic_level_end = semantic_label_vis(im_semantic_treelabel, dataset.colour_map_np_level)
            im_semantic_level_gt_end = semantic_label_vis(im_semantic_gt[:-1, :, :], dataset.colour_map_np_level)
            save_name = save_name_base+"_level{}".format(len(dataset.num_semantic)-2)
            print("===> save at: ", os.path.join(render_sem_dir,  save_name+".png"))
            print("===> save at: ", os.path.join(render_sem_dir,  save_name+"_gt.png"))
            cv2.imwrite(os.path.join(render_sem_dir,  save_name+".png"), im_semantic_level_end[...,::-1])
            cv2.imwrite(os.path.join(render_sem_dir,  save_name+"_gt.png"), im_semantic_level_gt_end[...,::-1])

        if flag_show_multilevel and flag_use_method==1:
            # use mlp transfer: im_semantic_label_mlp
            save_name_base = "sem_{:04d}".format(time_idx)
            label_colormap_full = label_colormap()

            # level-end
            im_semantic_level_end = im_semantic_label_mlp.detach().cpu().numpy()
            im_semantic_level_end = dataset.colors_map_all[im_semantic_level_end].astype(np.uint8) 
            im_semantic_level_end = np.squeeze(im_semantic_level_end) 
            im_semantic_level_gt_end = semantic_label_vis(im_semantic_gt[:-1, :, :], dataset.colour_map_np_level)
            label_map_vis = im_semantic_level_end[...,::-1]
            label_map_gt_vis = im_semantic_level_gt_end[...,::-1]
            save_name = save_name_base+"_level{}".format(len(dataset.num_semantic)-2)
            print("===> save at: ", os.path.join(render_sem_dir,  save_name+".png"))
            print("===> save at: ", os.path.join(render_sem_dir,  save_name+"_gt.png"))
            if flag_show_transparent==1:
                img_path = dataset.color_paths[time_idx]
                color = np.asarray(imageio.imread(img_path), dtype=float)
                if "scannet" in dataset.dataset_name:
                    color = cv2.resize(color, (640, 480), interpolation=cv2.INTER_LINEAR)
                label_map_show = cv2.addWeighted(color, w_color, label_map_vis, w_sem, 0, dtype=cv2.CV_8U)
                label_map_gt_show = cv2.addWeighted(color, w_color, label_map_gt_vis, w_sem, 0, dtype=cv2.CV_8U)
            elif flag_show_transparent==0:
                label_map_show = label_map_vis
                label_map_gt_show = label_map_gt_vis
            cv2.imwrite(os.path.join(render_sem_dir,  save_name+".png"), label_map_show)
            cv2.imwrite(os.path.join(render_sem_dir,  save_name+"_gt.png"), label_map_gt_show)

            # multi-level
            i_vis_end = 0
            for i_level in range(len(dataset.num_semantic)-2):
                # 1. data
                # 1.1 flat->i-level
                im_semantic_level_source = []
                for i_level_in in range(i_level+1):
                    im_semantic_level_source_tmp = transfer_back_eachlevel(im_semantic_label_mlp, dataset, i_level_in)
                    im_semantic_level_source.append(im_semantic_level_source_tmp)
                im_semantic_level_source = torch.stack(im_semantic_level_source, dim=0)
                # 1.2 each level
                im_semantic_level, num_dim_est = transfer_eachlevel_1(im_semantic_level_source, dataset)
                im_semantic_gt_level, num_dim_gt = transfer_eachlevel_1(im_semantic_gt[:i_level+1, :, :], dataset)
                num_dim = max(num_dim_est, num_dim_gt)
                # 2. color_map
                label_color_map = label_colormap_full[i_vis_end:i_vis_end+num_dim, :]
                label_map_vis = label_color_map[im_semantic_level]
                label_map_gt_vis = label_color_map[im_semantic_gt_level]
                label_map_vis = label_map_vis[...,::-1]
                label_map_gt_vis = label_map_gt_vis[...,::-1]
                i_vis_end += num_dim
                # (optional) transparent
                if flag_show_transparent==1:
                    img_path = dataset.color_paths[time_idx]
                    color = np.asarray(imageio.imread(img_path), dtype=float)
                    label_map_show = cv2.addWeighted(color, w_color, label_map_vis, w_sem, 0, dtype=cv2.CV_8U)
                    label_map_gt_show = cv2.addWeighted(color, w_color, label_map_gt_vis, w_sem, 0, dtype=cv2.CV_8U)
                    pass
                elif flag_show_transparent==0:
                    label_map_show = label_map_vis
                    label_map_gt_show = label_map_gt_vis
                # save
                save_name = save_name_base+"_level{}".format(i_level)
                print("===> save at: ", os.path.join(render_sem_dir,  save_name+".png"))
                print("===> save at: ", os.path.join(render_sem_dir,  save_name+"_gt.png"))
                cv2.imwrite(os.path.join(render_sem_dir,  save_name+".png"), label_map_show)
                cv2.imwrite(os.path.join(render_sem_dir,  save_name+"_gt.png"), label_map_gt_show)
            


        