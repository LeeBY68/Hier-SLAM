import os
import argparse
from importlib.machinery import SourceFileLoader
import shutil, re, time, sys

import numpy as np
import torch
from plyfile import PlyData, PlyElement
import torch.nn.functional as F
from imgviz import label_colormap

flag_mlp_use = 0        # 0: no mlp; 1: use mlp

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.insert(0, _BASE_DIR)

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
)

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
    else:
        raise ValueError(f"Unknown dataset name {config_dict['dataset_name']}")


# Spherical harmonic constant
C0 = 0.28209479177387814


# semantic visualization

nyu13_colour_code = (np.array([[0, 0, 0],
                       [0, 0, 1], # BED
                       [0.9137,0.3490,0.1882], #BOOKS
                       [0, 0.8549, 0], #CEILING
                       [0.5843,0,0.9412], #CHAIR
                       [0.8706,0.9451,0.0941], #FLOOR
                       [1.0000,0.8078,0.8078], #FURNITURE
                       [0,0.8784,0.8980], #OBJECTS
                       [0.4157,0.5333,0.8000], #PAINTING
                       [0.4588,0.1137,0.1608], #SOFA
                       [0.9412,0.1373,0.9216], #TABLE
                       [0,0.6549,0.6118], #TV
                       [0.9765,0.5451,0], #WALL
                       [0.8824,0.8980,0.7608]])*255).astype(np.uint8)   # window

# color palette for nyu34 labels
nyu34_colour_code = np.array([
       (0, 0, 0),
       (174, 199, 232),		# wall
       (152, 223, 138),		# floor
       (31, 119, 180), 		# cabinet
       (255, 187, 120),		# bed
       (188, 189, 34), 		# chair
       (140, 86, 75),  		# sofa
       (255, 152, 150),		# table
       (214, 39, 40),  		# door
       (197, 176, 213),		# window
    #    (148, 103, 189),		# bookshelf
       (196, 156, 148),		# picture
       (23, 190, 207), 		# counter
       (178, 76, 76),       # blinds
       (247, 182, 210),		# desk
       (66, 188, 102),      # shelves
       (219, 219, 141),		# curtain
    #    (140, 57, 197),    # dresser
       (202, 185, 52),      # pillow
    #    (51, 176, 203),    # mirror
       (200, 54, 131),      # floor
       (92, 193, 61),       # clothes
       (78, 71, 183),       # ceiling
       (172, 114, 82),      # books
       (255, 127, 14), 		# refrigerator
       (91, 163, 138),      # tv
       (153, 98, 156),      # paper
       (140, 153, 101),     # towel
    #    (158, 218, 229),		# shower curtain
       (100, 125, 154),     # box
    #    (178, 127, 135),       # white board
    #    (120, 185, 128),       # person
       (146, 111, 194),     # night stand
       (44, 160, 44),  		# toilet
       (112, 128, 144),		# sink
       (96, 207, 209),      # lamp
       (227, 119, 194),		# bathtub
       (213, 92, 176),      # bag
       (94, 106, 211),      # other struct
       (82, 84, 163),  		# otherfurn
       (100, 85, 144)       # other prop
    ]).astype(np.uint8)

# color palette for nyu40 labels
nyu40_colour_code = np.array([
       (0, 0, 0),

       (174, 199, 232),		# wall
       (152, 223, 138),		# floor
       (31, 119, 180), 		# cabinet
       (255, 187, 120),		# bed
       (188, 189, 34), 		# chair

       (140, 86, 75),  		# sofa
       (255, 152, 150),		# table
       (214, 39, 40),  		# door
       (197, 176, 213),		# window
       (148, 103, 189),		# bookshelf

       (196, 156, 148),		# picture
       (23, 190, 207), 		# counter
       (178, 76, 76),       # blinds
       (247, 182, 210),		# desk
       (66, 188, 102),      # shelves

       (219, 219, 141),		# curtain
       (140, 57, 197),    # dresser
       (202, 185, 52),      # pillow
       (51, 176, 203),    # mirror
       (200, 54, 131),      # floor

       (92, 193, 61),       # clothes
       (78, 71, 183),       # ceiling
       (172, 114, 82),      # books
       (255, 127, 14), 		# refrigerator
       (91, 163, 138),      # tv

       (153, 98, 156),      # paper
       (140, 153, 101),     # towel
       (158, 218, 229),		# shower curtain
       (100, 125, 154),     # box
       (178, 127, 135),       # white board

       (120, 185, 128),       # person
       (146, 111, 194),     # night stand
       (44, 160, 44),  		# toilet
       (112, 128, 144),		# sink
       (96, 207, 209),      # lamp

       (227, 119, 194),		# bathtub
       (213, 92, 176),      # bag
       (94, 106, 211),      # other struct
       (82, 84, 163),  		# otherfurn
       (100, 85, 144)       # other prop
    ]).astype(np.uint8)


def transfer_eachlevel_1(im_semantic_levels, dataset):
    level_num = im_semantic_levels.shape[0]
    idx_mapping = dataset.tree_id_classes_map[level_num-1]
    num_dim = len(idx_mapping)
    im_semantic_label = torch.full((im_semantic_levels.shape[1], 1), -1)

    for idx, (key, value) in enumerate(idx_mapping.items(), start=0):
        # print(key, value)
        key = torch.tensor(key)
        key = key.unsqueeze(1)
        index = (im_semantic_levels==key)      # [level_num, h, w] Fix: Comparing with the unsqueezed key tensor
        index = torch.all(index, dim=0)
        im_semantic_label[index] = idx
    
    im_semantic_label = im_semantic_label.squeeze(1)

    return im_semantic_label, num_dim

def rgb_to_spherical_harmonic(rgb):
    return (rgb-0.5) / C0


def spherical_harmonic_to_rgb(sh):
    return sh*C0 + 0.5

def transfer_tree_label(im_semantic, tree_num_semantic):
    """
    im_semantic: [N, num_semantic_embed]
    """
    im_semantic = torch.from_numpy(im_semantic)

    logits_2_label = lambda x: torch.argmax(torch.nn.functional.softmax(x, dim=-1),dim=-1)
        
    idx_begin = 0
    for i in range(len(tree_num_semantic)-1):
        num_classes_level = tree_num_semantic[i]
        idx_end = idx_begin + num_classes_level
        im_semantic_level = im_semantic[:, idx_begin:idx_end]
        semantic_label = logits_2_label(im_semantic_level)
        semantic_label = semantic_label.unsqueeze(0)
        if i == 0:
            im_semantic_treelabel = semantic_label
        else:
            im_semantic_treelabel = torch.cat((im_semantic_treelabel, semantic_label), dim=0)
        idx_begin = idx_end
    return im_semantic_treelabel

def semantic_label_vis(label_map_tree, colour_map_level):
    """
    label_map_tree:[3, N]
    """

    label_map_tree_vis = torch.zeros(label_map_tree.shape[1], 3, dtype=torch.uint8)
    for key, value in colour_map_level.items():
        key = key.unsqueeze(1)
        
        res = (label_map_tree==key)
        res = torch.all(res, dim=0)

        color_value = torch.from_numpy(value)

        label_map_tree_vis[res] = color_value
        
    label_map_tree_vis = np.array(label_map_tree_vis)
    
    return label_map_tree_vis


def save_ply(path, means, scales, rotations, rgbs, opacities, normals=None):
    if normals is None:
        normals = np.zeros_like(means)

    print(rgbs.min(), rgbs.max())
    colors = rgb_to_spherical_harmonic(rgbs)
    print(colors.min(), colors.max())

    if scales.shape[1] == 1:
        scales = np.tile(scales, (1, 3))

    attrs = ['x', 'y', 'z',
             'nx', 'ny', 'nz',
             'f_dc_0', 'f_dc_1', 'f_dc_2',
             'opacity',
             'scale_0', 'scale_1', 'scale_2',
             'rot_0', 'rot_1', 'rot_2', 'rot_3',]

    dtype_full = [(attribute, 'f4') for attribute in attrs]
    elements = np.empty(means.shape[0], dtype=dtype_full)

    attributes = np.concatenate((means, normals, colors, opacities, scales, rotations), axis=1)
    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(path)

    print(f"Saved PLY format Splat to {path}")

def save_ply_semantic(dataset, path, means, scales, rotations, 
                      rgbs, opacities, semantics, normals=None, flag_mlp = 0, mlp_func = None):
    if normals is None:
        normals = np.zeros_like(means)

    flag_mlp_show = 0   # 0: directly transform from multi-level tree; 1: use mlp transfer

    if flag_mlp_show == 1:
        # transfer numpy to tensor
        semantics = torch.from_numpy(semantics).cuda()
        im_semantic_in = semantics.unsqueeze(0).unsqueeze(0).permute(0, 3, 2, 1)             # [1, num_emdedding, h, w]]
        logits = mlp_func(im_semantic_in).squeeze()                                          # [num_classes, h, w]
        logits = F.softmax(logits, dim=0)
        im_semantic_label = torch.argmax(logits, dim=0)
        im_semantic_label = im_semantic_label.detach().cpu().numpy()
        semantic_map_label_colorbar = dataset.colors_map_all[im_semantic_label]
    elif flag_mlp_show == 0:
        semantics_treelabel = transfer_tree_label(semantics, dataset.num_semantic)
        semantics_treelabel = semantics_treelabel.detach().cpu()
        semantic_map_label_colorbar = semantic_label_vis(semantics_treelabel, dataset.colour_map_np_level)
    
    colors = semantic_map_label_colorbar
    use_num = colors.shape[0]
    if use_num < colors.shape[0]:
        colors = colors[:use_num, :]
        means = means[:use_num, :]
        normals = normals[:use_num, :]
        scales = scales[:use_num, :]
        rotations = rotations[:use_num, :]
        opacities = opacities[:use_num, :]

    if scales.shape[1] == 1:
        scales = np.tile(scales, (1, 3))

    dtype_full = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
             ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
             ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),
             ('opacity', 'f4'),
             ('scale_0', 'f4'), ('scale_1', 'f4'), ('scale_2', 'f4'),
             ('rot_0', 'f4'), ('rot_1', 'f4'), ('rot_2', 'f4'), ('rot_3', 'f4'),]

    elements = np.empty(means.shape[0], dtype=dtype_full)

    attributes = np.concatenate((means, normals, colors, opacities, scales, rotations), axis=1)
    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(path)

    print(f"Saved PLY format Splat to {path}")

def save_ply_semantic_multilevel(dataset, path, means, scales, rotations, 
                      rgbs, opacities, semantics, i_level, normals=None, flag_mlp = 0, mlp_func = None):
    if normals is None:
        normals = np.zeros_like(means)

    # original transfer
    flag_mlp_show = 0       # 0: directly transform from multi-level tree; 1: use mlp transfer

    if flag_mlp_show == 1:
        # transfer numpy to tensor
        semantics = torch.from_numpy(semantics).cuda()
        im_semantic_in = semantics.unsqueeze(0).unsqueeze(0).permute(0, 3, 2, 1)                           # [1, num_emdedding, h, w]]
        logits = mlp_func(im_semantic_in).squeeze()                        # [num_classes, h, w]
        logits = F.softmax(logits, dim=0)
        im_semantic_label = torch.argmax(logits, dim=0)
        im_semantic_label = im_semantic_label.detach().cpu().numpy()
        semantic_map_label_colorbar = dataset.colors_map_all[im_semantic_label]
    elif flag_mlp_show == 0:
        semantics_treelabel = transfer_tree_label(semantics, dataset.num_semantic )
        # level info
        sem_level = semantics_treelabel[:i_level+1, :]
        im_semantic_level, num_dim = transfer_eachlevel_1(sem_level, dataset)
        label_color_map = label_colormap(num_dim)
        semantic_map_label_colorbar = label_color_map[im_semantic_level]
    
    colors = semantic_map_label_colorbar
    # use_num = 700000
    use_num = colors.shape[0]
    if use_num < colors.shape[0]:
        colors = colors[:use_num, :]
        means = means[:use_num, :]
        normals = normals[:use_num, :]
        scales = scales[:use_num, :]
        rotations = rotations[:use_num, :]
        opacities = opacities[:use_num, :]

    if scales.shape[1] == 1:
        scales = np.tile(scales, (1, 3))

    dtype_full = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
             ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
             ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),
             ('opacity', 'f4'),
             ('scale_0', 'f4'), ('scale_1', 'f4'), ('scale_2', 'f4'),
             ('rot_0', 'f4'), ('rot_1', 'f4'), ('rot_2', 'f4'), ('rot_3', 'f4'),]

    elements = np.empty(means.shape[0], dtype=dtype_full)

    attributes = np.concatenate((means, normals, colors, opacities, scales, rotations), axis=1)
    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(path)

    print(f"Saved PLY format Splat to {path}")



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="Path to config file.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Load config
    experiment = SourceFileLoader(os.path.basename(args.config), args.config).load_module()
    config = experiment.config
    work_path = config['workdir']
    run_name = config['run_name']
    params_path = os.path.join(work_path, run_name, "params.npz")
    print('params_path is: ', params_path)

    params = dict(np.load(params_path, allow_pickle=True))
    print(params.keys())
    means = params['means3D']
    scales = params['log_scales']
    rotations = params['unnorm_rotations']
    rgbs = params['rgb_colors']
    opacities = params['logit_opacities']
    semantics = params['semantic']

    name_save = "hierslam_mlp{}.ply".format(flag_mlp_use)
    ply_path = os.path.join(work_path, run_name, name_save)

    # dataset =====
    results_dir = os.path.join(
        experiment.config["workdir"], experiment.config["run_name"]
    )
    experiment.config["results_dir"] = results_dir

    print("Loading Dataset ...")
    dataset_config = config["data"]
    if "gradslam_data_cfg" not in dataset_config:
        gradslam_data_cfg = {}
        gradslam_data_cfg["dataset_name"] = dataset_config["dataset_name"]
    else:
        gradslam_data_cfg = load_dataset_config(dataset_config["gradslam_data_cfg"])
        gradslam_data_cfg = {**gradslam_data_cfg, **dataset_config}
    gradslam_data_cfg["results_dir"] = results_dir
    if "ignore_bad" not in dataset_config:
        dataset_config["ignore_bad"] = False
    if "use_train_split" not in dataset_config:
        dataset_config["use_train_split"] = True
    
    # Poses are relative to the first training frame
    dataset = get_dataset(
        config_dict=gradslam_data_cfg,
        basedir=dataset_config["basedir"],
        sequence=os.path.basename(dataset_config["sequence"]),
        start=dataset_config["start"],
        end=dataset_config["end"],
        stride=dataset_config["stride"],
        desired_height=dataset_config["desired_image_height"],
        desired_width=dataset_config["desired_image_width"],
        device=torch.device("cuda"),
        relative_pose=True,
        ignore_bad=dataset_config["ignore_bad"],
        use_train_split=dataset_config["use_train_split"],
    )
    num_frames = dataset_config["num_frames"]
    if num_frames == -1:
        num_frames = len(dataset)
    semantic_dim = sum(dataset.num_semantic[:-1])
    print("semantic dim is: ", semantic_dim)
    # dataset =====

    # ==== mlp ====
    if flag_mlp_use == 1:
        MLP_func = torch.nn.Conv2d(semantic_dim, dataset.num_semantic_class, kernel_size=1)
        MLP_func.load_state_dict(torch.load(results_dir+'/Semantic.pth'))
        MLP_func.cuda()
        MLP_func.eval()
    elif flag_mlp_use == 0:
        MLP_func = None

    save_ply_semantic(dataset, ply_path, means, scales, rotations, rgbs, opacities, semantics, flag_mlp = flag_mlp_use, mlp_func = MLP_func)
    # save_ply_semantic_multilevel(dataset, ply_path, means, scales, rotations, rgbs, opacities, semantics, level_use,flag_mlp = flag_mlp_use, mlp_func = MLP_func)