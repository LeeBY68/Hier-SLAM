import glob
import os
from pathlib import Path
from typing import Dict, List, Optional, Union

import csv
from tqdm import tqdm
import imageio
import cv2
import imgviz
from imgviz import label_colormap
from imgviz import draw as draw_module
import matplotlib.pyplot as plt

import numpy as np
import torch
from natsort import natsorted
import matplotlib.cm as cm

from .geometryutils import relative_transformation
from . import datautils

from .basedataset import GradSLAMDataset

# os.environ['CUDA_VISIBLE_DEVICES'] = "0"
class ScannetDataset(GradSLAMDataset):
    def __init__(
        self,
        config_dict,
        basedir,
        sequence,
        stride: Optional[int] = None,
        start: Optional[int] = 0,
        end: Optional[int] = -1,
        desired_height: Optional[int] = 968,
        desired_width: Optional[int] = 1296,
        load_embeddings: Optional[bool] = False,
        embedding_dir: Optional[str] = "embeddings",
        embedding_dim: Optional[int] = 512,
        **kwargs,
    ):
        self.input_folder = os.path.join(basedir, sequence)
        self.pose_path = None
        super().__init__(
            config_dict,
            stride=stride,
            start=start,
            end=end,
            desired_height=desired_height,
            desired_width=desired_width,
            load_embeddings=load_embeddings,
            embedding_dir=embedding_dir,
            embedding_dim=embedding_dim,
            **kwargs,
        )

    def get_filepaths(self):
        color_paths = natsorted(glob.glob(f"{self.input_folder}/color/*.jpg"))
        depth_paths = natsorted(glob.glob(f"{self.input_folder}/depth/*.png"))
        embedding_paths = None
        if self.load_embeddings:
            embedding_paths = natsorted(glob.glob(f"{self.input_folder}/{self.embedding_dir}/*.pt"))
        return color_paths, depth_paths, embedding_paths

    def load_poses(self):
        poses = []
        posefiles = natsorted(glob.glob(f"{self.input_folder}/pose/*.txt"))
        for posefile in posefiles:
            _pose = torch.from_numpy(np.loadtxt(posefile))
            poses.append(_pose)
        return poses

    def read_embedding_from_file(self, embedding_file_path):
        print(embedding_file_path)
        embedding = torch.load(embedding_file_path, map_location="cpu")
        return embedding.permute(0, 2, 3, 1)  # (1, H, W, embedding_dim)

class ScannetDataset_semantic(GradSLAMDataset):
    def __init__(
        self,
        config_dict,
        basedir,
        sequence,
        stride: Optional[int] = None,
        start: Optional[int] = 0,
        end: Optional[int] = -1,
        desired_height: Optional[int] = 968,
        desired_width: Optional[int] = 1296,
        load_embeddings: Optional[bool] = False,
        embedding_dir: Optional[str] = "embeddings",
        embedding_dim: Optional[int] = 512,
        **kwargs,
    ):
        # temp setting ======================
        self.flag_load_semantic = False
        flag_save = True
        # to save time, set [flag_save = True & self.flag_load_semantic = False] for the first time running. 
        # after that, set [flag_save = False & self.flag_load_semantic = True] to read semantic hierarchy directly from saved path.

        # 1. Initialize parameters ========== 
        Flag_StoreLabel = True
        self.input_folder = os.path.join(basedir, sequence)
        self.pose_path = None
        self.results_dir = config_dict["results_dir"]
        self.use_pyramid = config_dict["use_pyramid"]
        self.pyramid_level = config_dict["pyramid_level"]
        self.dataset_name = config_dict["dataset_name"]
        
        vis_label_save_dir = os.path.join(self.input_folder, "vis_semantic_label")
        if not os.path.exists(vis_label_save_dir):
            os.makedirs(vis_label_save_dir, exist_ok=True)

        # semantic info load 
        self.label_paths = natsorted(glob.glob(f"{self.input_folder}/label-filt/*.png"))
        self.sem_mode = config_dict["sem_mode"]
        print("     semantic mode is: ", self.sem_mode)
        if start<end and start>=0:
            self.label_paths = self.label_paths[start : end : stride]

        if self.sem_mode=="nyu40":  # use nyu40 provided by official scannet
            label_mapping_nyu = self.load_scannet_nyu40_mapping(self.input_folder)
            colour_map_np = nyu40_colour_code
            self.num_semantic = 41
            class_name_string = ["void","wall", "floor", "cabinet", "bed", "chair",
                                "sofa", "table", "door", "window", "bookshelf", 
                                "picture", "counter", "blinds", "desk", "shelves",
                                "curtain", "dresser", "pillow", "mirror", "floor",
                                "clothes", "ceiling", "books", "fridge", "tv",
                                "paper", "towel", "shower curtain", "box", "white board",
                                "person", "night stand", "toilet", "sink", "lamp",
                                "bath tub", "bag", "other struct", "other furniture", "other prop"]
            assert colour_map_np.shape[0] == 41
        elif self.sem_mode=="tree":
            # 1. tree-level read
            label_mapping_tree, label_mapping_tree_label, tree_id_classes_map = self.load_scannet_tree_mapping_nyu40level4(self.input_folder, 'scannetv2-labels.combined.tree.tsv')
            max_level1, max_level2, max_level3, max_level4 = find_max_level4(label_mapping_tree)  
            
            # 2. tree: based on nyu40, so also read nyu40 color mapping
            # class_name_string: same as nyu40-id order
            label_mapping_nyu = self.load_scannet_nyu40_mapping(self.input_folder)
            colour_map_np = nyu40_colour_code
            class_name_string = ["void","wall", "floor", "cabinet", "bed", "chair",
                                "sofa", "table", "door", "window", "bookshelf", 
                                "picture", "counter", "blinds", "desk", "shelves",
                                "curtain", "dresser", "pillow", "mirror", "floor",
                                "clothes", "ceiling", "books", "fridge", "tv",
                                "paper", "towel", "shower curtain", "box", "white board",
                                "person", "night stand", "toilet", "sink", "lamp",
                                "bath tub", "bag", "other struct", "other furniture", "other prop"]
            
            # semantic embedding number for each level
            self.num_semantic = [max_level1+1, max_level2+1, max_level3+1, max_level4+1, len(class_name_string)]

            # 3. tree-level visualization
            """
            label_mapping_tree: dict (len=40), {nyu40_id: [level1_id, level2_id, level3_id]}
            label_mapping_tree_label: dict (len=40), {[nyu40_class]: [level1_class, level2_class, level3_class]}
            tree_id_classes_map [level0, level1, level2, original-level]: each level has a dict, '[class_id, ...]': ['class', ...]
            label_mapping_nyu: [original_id: nyu40_id] -> len = 549
            self.colour_map_np_level: len=40 {[class_id1, class_id2, class_id3]: [r,g,b]}
            """
            self.colors_map_all = colour_map_np
            tree_id_classes_map_leaf = tree_id_classes_map[-2]      # obtain the finest level class
            self.colour_map_np_level_map = tree_id_classes_map_leaf
            self.colour_map_np_level = {}
            self.label_mapping_tree = label_mapping_tree

            # get colormap for visualization
            for idx, (key, value) in enumerate(tree_id_classes_map_leaf.items(), 0):
                key = torch.tensor(key)
                self.colour_map_np_level[key] = self.colors_map_all[idx, :]

            # save colorbar (optional)
            classes = [i for i in range(len(tree_id_classes_map_leaf)) ]
            classes_name = [ (str(key)+": "+str(value)) for key, value in tree_id_classes_map_leaf.items() ]
            legend_tree = plot_semantic_legend(classes, classes_name, 
                        colormap = self.colors_map_all, save_path=vis_label_save_dir, save_name="semantic_class_Legend_leaf_nyu40_4")
            legend_tree = plot_semantic_legend(classes, classes_name, 
                        colormap =  self.colors_map_all, save_path=self.results_dir, save_name="semantic_class_Legend_leaf_nyu40_4")
            tmp = np.arange(0, colour_map_np.shape[0])
            legend_tree = plot_semantic_legend(np.arange(0, colour_map_np.shape[0]), class_name_string, 
                        colormap = colour_map_np, save_path=vis_label_save_dir, save_name="semantic_class_Legend_nyu40")
            legend_tree = plot_semantic_legend(np.arange(0, colour_map_np.shape[0]), class_name_string, 
                        colormap =  colour_map_np, save_path=self.results_dir, save_name="semantic_class_Legend_nyu40")
        # elif self.sem_mode=="ab_tree_level5_nyu40": # if OK, delete
        elif self.sem_mode=="tree_large":
            # 1. tree-level read
            # name_read = 'scannetv2-labels.combined.tree_raw_level5_fromnyu40.tsv' # if OK, delete
            name_read = 'scannetv2-labels.combined.tree-large.tsv'
            label_mapping_tree, label_mapping_tree_label, tree_id_classes_map = self.load_scannet_tree_mapping_rawtolevel5(self.input_folder, name_read) 
                
            num_level_list = find_max_level(label_mapping_tree, flag_add=True)
            num_level_list.append(len(label_mapping_tree)) 
            self.num_semantic = num_level_list

            self.colors_map_all = imgviz.label_colormap(self.num_semantic[-1])
            colour_map_np = self.colors_map_all
            self.colour_map_np = self.colors_map_all
            tree_id_classes_map_leaf = tree_id_classes_map[-2]      # obtain the finest level class
            self.colour_map_np_level_map = tree_id_classes_map_leaf
            self.colour_map_np_level = {}
            self.label_mapping_tree = label_mapping_tree
            self.tree_id_classes_map = tree_id_classes_map

            # get colormap for visualization
            for idx, (key, value) in enumerate(tree_id_classes_map_leaf.items(), 0):
                key = torch.tensor(key)
                self.colour_map_np_level[key] = self.colors_map_all[idx, :]
            # raw class 
            raw_class_name_string = []
            raw_class_name_id = []
            for key, item in tree_id_classes_map[-1].items():
                raw_class_name_string.append(item)
                raw_class_name_id.append(key)
            self.semantic_class = raw_class_name_string
            self.semantic_id = raw_class_name_id
            
            # save colorbar (optional)
            classes = [i for i in range(len(tree_id_classes_map_leaf)) ]
            classes_name = [ (str(key)+": "+str(value)) for key, value in tree_id_classes_map_leaf.items() ]
            visual_semantic_legend(classes, classes_name, self.colors_map_all, self.results_dir, "semantic_class_Legend_leaf")
            visual_semantic_legend(classes, classes_name, self.colors_map_all, vis_label_save_dir, "semantic_class_Legend_leaf")
            visual_semantic_legend(np.arange(0, self.colors_map_all.shape[0]), raw_class_name_string, self.colors_map_all, self.results_dir, "semantic_class_Legend_base")
            visual_semantic_legend(np.arange(0, self.colors_map_all.shape[0]), raw_class_name_string, self.colors_map_all, vis_label_save_dir, "semantic_class_Legend_base")
            pass
        else:
            assert False

        self.num_semantic_class = colour_map_np.shape[0]

        # basic info load
        super().__init__(
            config_dict,
            stride=stride,
            start=start,
            end=end,
            desired_height=desired_height,
            desired_width=desired_width,
            load_embeddings=load_embeddings,
            embedding_dir=embedding_dir,
            embedding_dim=embedding_dim,
            **kwargs,
        )
        assert(len(self.label_paths)==len(self.color_paths))
        assert(len(self.label_paths)==self.num_imgs)
        
        # semantic only read path ========
        if self.flag_load_semantic:
            if self.sem_mode=="nyu40":
                self.load_sem_path = os.path.join(self.input_folder, "semantic_labels_nyu40")
                self.colour_map_np = colour_map_np
                for i in range(len(self.label_paths)):
                    self.label_paths[i] = self.label_paths[i].replace("label-filt", "semantic_labels_nyu40").replace("png", "npy")
            if self.sem_mode=="tree":
                # self.load_sem_path = os.path.join(self.input_folder, "semantic_labels_tree")      # use this
                name_use = "semantic_labels_nyu40tree4level_5"
                self.load_sem_path = os.path.join(self.input_folder, name_use)
                self.colour_map_np = colour_map_np
                for i in range(len(self.label_paths)):
                    self.label_paths[i] = self.label_paths[i].replace("label-filt", name_use).replace("png", "npy")
            if self.sem_mode=="tree_large":
                # name_use = "semantic_labels_tree_large"           # use this  
                name_use = "semantic_labels_ab_LLM_0_tree5level_nyu40"      
                self.load_sem_path = os.path.join(self.input_folder, name_use)
                self.colour_map_np = colour_map_np
                for i in range(len(self.label_paths)):
                    self.label_paths[i] = self.label_paths[i].replace("label-filt", name_use).replace("png", "npy")
        # ================================

        # 2. Semantic info save/load ========== 
        if not self.flag_load_semantic:
            vis_sem_stride = 1
            semantic_raw = []
            for item in tqdm(range(self.num_imgs)):
                # load semantic raw info
                label = np.asarray(imageio.imread(self.label_paths[item]), dtype=np.int64)
                # resize
                label = self._preprocess_semantic_label(label)
                #add list
                semantic_raw.append(label)

            semantic_raw = np.asarray(semantic_raw) # [num_img, h, w]

            # [id transformarion] 1. transform to nyu-40: semantic_raw_nyu
            if "ab" in self.sem_mode:
                semantic_raw_nyu = semantic_raw.copy()
            else:
                semantic_raw_nyu = semantic_raw.copy()
                for scan_id, nyu_id in label_mapping_nyu.items():
                    semantic_raw_nyu[semantic_raw==scan_id] = nyu_id
                # semantic_raw_nyu: [num_img, h, w, 1]

            # [id transformarion] - 2. transform to tree level
            if self.sem_mode == "eigen13" or self.sem_mode == "nyu40" or self.sem_mode == "raw":
                self.semantic_labels_raw_gt = semantic_raw_nyu
            elif self.sem_mode == "tree":       # 4 levels read
                semantic_raw_tree_all = []
                semantic_raw_tree_level0 = semantic_raw_nyu.copy()
                semantic_raw_tree_level1 = semantic_raw_nyu.copy()
                semantic_raw_tree_level2 = semantic_raw_nyu.copy()
                semantic_raw_tree_level3 = semantic_raw_nyu.copy()
                for scan_id, tree_id in label_mapping_tree.items():
                    # print(scan_id, tree_id)
                    semantic_raw_tree_level0[semantic_raw_nyu==scan_id] = tree_id[0]
                    semantic_raw_tree_level1[semantic_raw_nyu==scan_id] = tree_id[1]
                    semantic_raw_tree_level2[semantic_raw_nyu==scan_id] = tree_id[2]
                    semantic_raw_tree_level3[semantic_raw_nyu==scan_id] = tree_id[3]

                semantic_raw_tree_all.append(semantic_raw_tree_level0.squeeze())
                semantic_raw_tree_all.append(semantic_raw_tree_level1.squeeze())
                semantic_raw_tree_all.append(semantic_raw_tree_level2.squeeze())
                semantic_raw_tree_all.append(semantic_raw_tree_level3.squeeze())
                semantic_raw_tree_all.append(semantic_raw_nyu.squeeze())          # original nyu40
                semantic_raw_tree_all = np.asarray(semantic_raw_tree_all)
                self.semantic_labels_raw_gt =semantic_raw_tree_all              # [level0, level1, level2, level3, original_nyu40]
            # print(self.semantic_labels_raw_gt.shape)    # [level_num, num_img, h, w]
            elif self.sem_mode=="tree_large":
                # 5-level read
                semantic_raw_tree_all = []
                semantic_raw_tree_level0 = semantic_raw_nyu.copy()
                semantic_raw_tree_level1 = semantic_raw_nyu.copy()
                semantic_raw_tree_level2 = semantic_raw_nyu.copy()
                semantic_raw_tree_level3 = semantic_raw_nyu.copy()
                semantic_raw_tree_level4 = semantic_raw_nyu.copy()
                for scan_id, tree_id in label_mapping_tree.items():
                    semantic_raw_tree_level0[semantic_raw_nyu==scan_id] = tree_id[0]
                    semantic_raw_tree_level1[semantic_raw_nyu==scan_id] = tree_id[1]
                    semantic_raw_tree_level2[semantic_raw_nyu==scan_id] = tree_id[2]
                    semantic_raw_tree_level3[semantic_raw_nyu==scan_id] = tree_id[3]
                    semantic_raw_tree_level4[semantic_raw_nyu==scan_id] = tree_id[4]

                semantic_raw_tree_all.append(semantic_raw_tree_level0.squeeze())
                semantic_raw_tree_all.append(semantic_raw_tree_level1.squeeze())
                semantic_raw_tree_all.append(semantic_raw_tree_level2.squeeze())
                semantic_raw_tree_all.append(semantic_raw_tree_level3.squeeze())
                semantic_raw_tree_all.append(semantic_raw_tree_level4.squeeze())
                semantic_raw_tree_all.append(semantic_raw_nyu.squeeze())          # original nyu40
                semantic_raw_tree_all = np.asarray(semantic_raw_tree_all)
                self.semantic_labels_raw_gt =semantic_raw_tree_all              # [level0, level1, level2, level3, original_nyu40]

                print(self.semantic_labels_raw_gt.shape)    # [level_num, num_img, h, w]

            if self.sem_mode == "nyu40":
                # concact classes
                self.semantic_classes = np.unique(self.semantic_labels_raw_gt).astype(np.uint8)
                self.num_semantic_class_compact = self.semantic_classes.shape[0]
                print("semantic classes number is: ", self.semantic_classes.shape)
                print("semantic raw class are: ", self.semantic_classes)
                # each scene may not contain all classes

                # label visualization
                colour_map_np_remap = colour_map_np.copy()[self.semantic_classes] # take corresponding colour map
                self.colour_map_np = colour_map_np                  # raw semantic mode color map
                self.colour_map_np_remap = colour_map_np_remap      # concact color map
                
                # save data, no need for read data again ==============================
                if flag_save:
                    # if save path not exist, mkdir it
                    if self.sem_mode=="nyu40":
                        sem_file_save_path = os.path.join(self.input_folder, 'semantic_labels_nyu40')
                    if self.sem_mode=="raw":
                        sem_file_save_path = os.path.join(self.input_folder, 'semantic_labels_rawnewid')
                    os.makedirs(sem_file_save_path, exist_ok=True)
                    
                    np.save(os.path.join(sem_file_save_path, "semantic_labels_raw_gt.npy"), self.semantic_labels_raw_gt)
                    for i in range(0, self.num_imgs):
                        save_name = os.path.splitext(os.path.basename(self.label_paths[i]))[0]
                        np.save(os.path.join(sem_file_save_path, "{}.npy".format(save_name)), self.semantic_labels_raw_gt[i, :, :, :])
                # =====================================================================

                # save colourised ground truth label to img folder
                if Flag_StoreLabel:
                    # save colourised ground truth label to img folder
                    vis_label_save_dir = os.path.join(self.input_folder, "vis_semantic_label_{}".format(self.sem_mode))
                    if not os.path.exists(vis_label_save_dir):
                        os.makedirs(vis_label_save_dir, exist_ok=True)
                    vis_train_label = colour_map_np[self.semantic_labels_raw_gt]
                    for i in range(0, self.num_imgs, vis_sem_stride):
                        label = vis_train_label[i].astype(np.uint8)     
                        label = np.squeeze(label)                   # [h,w,3]
                        cv2.imwrite(os.path.join(vis_label_save_dir, "vis_sem_{}.png".format(i)),label[...,::-1]) # bgr -> rgb  
                    # show semantic color bar
                    legend_img_arr_compact = plot_semantic_legend(self.semantic_classes, class_name_string, 
                                    colormap=colour_map_np, save_path=vis_label_save_dir, save_name="semantic_class_Legend_compact")
                    legend_img_arr_original = plot_semantic_legend(np.arange(0, self.num_semantic_class), class_name_string, 
                                    colormap=colour_map_np, save_path=vis_label_save_dir, save_name="semantic_class_Legend")
            else:   # tree
                self.colour_map_np = colour_map_np
                # tree level data (semantic gt) save
                if flag_save:
                    if self.sem_mode=="tree":
                        # sem_file_save_path = os.path.join(self.input_folder, 'semantic_labels_tree')  # use this
                        sem_file_save_path = os.path.join(self.input_folder, 'semantic_labels_nyu40tree4level_5')
                    elif self.sem_mode=="tree_large":
                        sem_file_save_path = os.path.join(self.input_folder, 'semantic_labels_tree_large')
                    os.makedirs(sem_file_save_path, exist_ok=True)
                    
                    np.save(os.path.join(sem_file_save_path, "semantic_labels_raw_gt.npy"), self.semantic_labels_raw_gt)
                    for i in range(0, self.num_imgs):
                        save_name = os.path.splitext(os.path.basename(self.label_paths[i]))[0]
                        np.save(os.path.join(sem_file_save_path, "{}.npy".format(save_name)), self.semantic_labels_raw_gt[:, i, :, :])
        # ================================

    def get_filepaths(self):
        color_paths = natsorted(glob.glob(f"{self.input_folder}/color/*.jpg"))
        depth_paths = natsorted(glob.glob(f"{self.input_folder}/depth/*.png"))
        embedding_paths = None
        if self.load_embeddings:
            embedding_paths = natsorted(glob.glob(f"{self.input_folder}/{self.embedding_dir}/*.pt"))
        return color_paths, depth_paths, embedding_paths

    def load_poses(self):
        poses = []
        posefiles = natsorted(glob.glob(f"{self.input_folder}/pose/*.txt"))
        for posefile in posefiles:
            _pose = torch.from_numpy(np.loadtxt(posefile))
            poses.append(_pose)
        return poses
    
    def __getitem__(self, index):
        # path
        color_path = self.color_paths[index]
        depth_path = self.depth_paths[index]
        color = np.asarray(imageio.imread(color_path), dtype=float)
        color = self._preprocess_color(color)

        if ".png" in depth_path:
            depth = np.asarray(imageio.imread(depth_path), dtype=np.int64)
        elif ".exr" in depth_path:
            depth = readEXR_onlydepth(depth_path)
        # load semantic info
        if self.flag_load_semantic:
            label = np.load(self.label_paths[index])
        else:
            # load array (only eigen13/nyu40/tree use save module)
            if self.sem_mode == "eigen13" or self.sem_mode == "nyu40":
                label = self.semantic_labels_raw_gt[index]
            else:   # tree
                label = self.semantic_labels_raw_gt[:, index, :, :]        # [level_num, num_img, h, w]
                label = label.squeeze()                                    # label: [level_num, h, w]

        K = as_intrinsics_matrix([self.fx, self.fy, self.cx, self.cy])
        if self.distortion is not None:
            # undistortion is only applied on color image, not depth!
            color = cv2.undistort(color, K, self.distortion)

        # 1. color & color pyramid
        if self.use_pyramid:
            pyramid = [color]
            color_py = color
            for i in range(1, self.pyramid_level):
                color_py = cv2.pyrDown(color_py)
                pyramid.append(color_py)
            pyramid = [torch.from_numpy(img) for img in pyramid]

        color = torch.from_numpy(color)
        K = torch.from_numpy(K)
        depth = self._preprocess_depth(depth)

        # 2. depth & depth pyramid
        if self.use_pyramid:
            pyramid_depth = [depth]
            depth_py = depth
            for i in range(1, self.pyramid_level):
                depth_py = cv2.pyrDown(depth_py)
                pyramid_depth.append(torch.from_numpy(depth_py).unsqueeze(2))
            pyramid_depth[0] = torch.from_numpy(pyramid_depth[0])
        depth = torch.from_numpy(depth)

        # 3. semantic & semantic label pyramid (semantc level: inverse of image level)
        if self.use_pyramid and self.sem_mode=="tree_nyu40_level3":
            assert(label.shape[0]==self.pyramid_level)

        if self.use_pyramid:
            pyramid_semantic = [label[-1, :, :]]
            for i in range(1, self.pyramid_level):
                idx = self.pyramid_level-1-i
                img_py = label[idx, :, :].squeeze()
                for i_py in range(i):
                    img_py = cv2.pyrDown(img_py.astype(np.float32))
                pyramid_semantic.append(torch.from_numpy(img_py).unsqueeze(2))
            pyramid_semantic[0] = torch.from_numpy(pyramid_semantic[0])
        # semantic info
        label = torch.from_numpy(label)

        K = datautils.scale_intrinsics(K, self.height_downsample_ratio, self.width_downsample_ratio)
        intrinsics = torch.eye(4).to(K)
        intrinsics[:3, :3] = K

        pose = self.transformed_poses[index]

        if self.load_embeddings:
            embedding = self.read_embedding_from_file(self.embedding_paths[index])
            if self.use_pyramid:
                return (
                    color.to(self.device).type(self.dtype),
                    depth.to(self.device).type(self.dtype),
                    intrinsics.to(self.device).type(self.dtype),
                    pose.to(self.device).type(self.dtype),
                    label.to(self.device).type(self.dtype),
                    [img.to(self.device).type(self.dtype) for img in pyramid],
                    [img_depth.to(self.device).type(self.dtype) for img_depth in pyramid_depth],
                    [img_sem.to(self.device).type(self.dtype) for img_sem in pyramid_semantic],
                    embedding.to(self.device),  # Allow embedding to be another dtype
                    # self.retained_inds[index].item(),
                )
            else:
                return (
                    color.to(self.device).type(self.dtype),
                    depth.to(self.device).type(self.dtype),
                    intrinsics.to(self.device).type(self.dtype),
                    pose.to(self.device).type(self.dtype),
                    label.to(self.device).type(self.dtype),
                    embedding.to(self.device),  # Allow embedding to be another dtype
                    # self.retained_inds[index].item(),
                )

        if self.use_pyramid:
            return (
                color.to(self.device).type(self.dtype),
                depth.to(self.device).type(self.dtype),
                intrinsics.to(self.device).type(self.dtype),
                pose.to(self.device).type(self.dtype),
                label.to(self.device).type(self.dtype),
                [img.to(self.device).type(self.dtype) for img in pyramid],
                [img_depth.to(self.device).type(self.dtype) for img_depth in pyramid_depth],
                [img_sem.to(self.device).type(self.dtype) for img_sem in pyramid_semantic],
                # self.retained_inds[index].item(),
            )
        else:
            return (
                color.to(self.device).type(self.dtype),
                depth.to(self.device).type(self.dtype),
                intrinsics.to(self.device).type(self.dtype),
                pose.to(self.device).type(self.dtype),
                label.to(self.device).type(self.dtype),
                # self.retained_inds[index].item(),
            )

    def read_embedding_from_file(self, embedding_file_path):
        print(embedding_file_path)
        embedding = torch.load(embedding_file_path, map_location="cpu")
        return embedding.permute(0, 2, 3, 1)  # (1, H, W, embedding_dim)
    
    def load_scannet_label_mapping(self, path):
        """ Returns a dict mapping scannet category label strings to scannet Ids

        scene****_**.aggregation.json contains the category labels as strings 
        so this maps the strings to the integer scannet Id

        Args:
            path: Path to the original scannet data.
                This is used to get scannetv2-labels.combined.tsv

        Returns:
            mapping: A dict from strings to ints
                example:
                    {'wall': 1,
                    'chair: 2,
                    'books': 22}

        """

        mapping = {}
        with open(os.path.join(path, 'scannetv2-labels.combined.tsv')) as tsvfile:
            tsvreader = csv.reader(tsvfile, delimiter='\t')
            for i, line in enumerate(tsvreader):
                if i==0:
                    continue
                scannet_id, name = int(line[0]), line[1]
                mapping[name] = scannet_id

        return mapping

    def load_scannet_nyu40_mapping(self, path):
        """ Returns a dict mapping scannet Ids to NYU40 Ids

        Args:
            path: Path to the original scannet data. 
                This is used to get scannetv2-labels.combined.tsv

        Returns:
            mapping: A dict from ints to ints
                example:
                    {1: 1,
                    2: 5,
                    22: 23}

        """

        mapping = {}
        with open(os.path.join(path, 'scannetv2-labels.combined.tsv')) as tsvfile:
            tsvreader = csv.reader(tsvfile, delimiter='\t')
            for i, line in enumerate(tsvreader):
                if i==0:
                    continue
                scannet_id, nyu40id = int(line[0]), int(line[4])
                mapping[scannet_id] = nyu40id
        return mapping

    def load_scannet_nyu13_mapping(self, path):
        """ Returns a dict mapping scannet Ids to NYU40 Ids

        Args:
            path: Path to the original scannet data. 
                This is used to get scannetv2-labels.combined.tsv

        Returns:
            mapping: A dict from ints to ints
                example:
                    {1: 1,
                    2: 5,
                    22: 23}

        """

        mapping = {}
        with open(os.path.join(path, 'scannetv2-labels.combined.tsv')) as tsvfile:
            tsvreader = csv.reader(tsvfile, delimiter='\t')
            for i, line in enumerate(tsvreader):
                if i==0:
                    continue
                scannet_id, nyu40id = int(line[0]), int(line[5])
                mapping[scannet_id] = nyu40id
        return mapping
    
    def load_scannet_binary_mapping(self, path):
        """ Returns a dict mapping scannet Ids to NYU40 Ids

        Args:
            path: Path to the original scannet data. 
                This is used to get scannetv2-labels.combined.tsv

        Returns:
            mapping: A dict from ints to ints
                example:
                    {1: 1,
                    2: 5,
                    22: 23}

        """

        mapping = {}
        with open(os.path.join(path, 'scannetv2-labels.combined.tsv')) as tsvfile:
            tsvreader = csv.reader(tsvfile, delimiter='\t')
            for i, line in enumerate(tsvreader):
                if i==0:
                    continue
                scannet_id, nyu40id = int(line[0]), int(line[5])
                if not nyu40id == 0:
                    mapping[scannet_id] = 1
                else:
                    mapping[scannet_id] = nyu40id
        return mapping
    
    def load_scannet_tree_mapping_nyu40level3(self, path):
        """ Returns a dict mapping scannet Ids to NYU40 Ids

        Args:
            path: Path to the original scannet data. 
                This is used to get scannetv2-labels.combined.tsv

        Returns:
            mapping: A dict from ints to ints
                example:
                    {1: 1,
                    2: 5,
                    22: 23}

        """
        mapping = {}
        mapping_label = {}
        
        with open(os.path.join(os.path.dirname(path), 'scannetv2-labels.combined.tree_nyu40_3leveltree.tsv')) as tsvfile:
            tsvreader = csv.reader(tsvfile, delimiter='\t')
            tree_level1_map = {}
            tree_level2_map = {}
            tree_level3_map = {}
            tree_original_map = {}
            for i, line in enumerate(tsvreader):
                if i==0:
                    continue
                if 17>len(line)-1 or line[17]=="":
                    level1_id = None
                    level1_id_label = None
                else:
                    level1_id = int(line[17])
                    level1_id_label = line[18]
                    tree_level1_map[level1_id] = level1_id_label
                if 19>len(line)-1 or line[19]=="":
                    level2_id = None
                    level2_id_label = None
                else:
                    level2_id = int(line[19])
                    level2_id_label = line[20]
                    tree_level2_map[level1_id, level2_id] = [level1_id_label, level2_id_label]
                if 21>len(line)-1 or line[21]=="":
                    level3_id = None
                    level3_id_label = None
                else:
                    level3_id = int(line[21])
                    level3_id_label = line[22]
                    tree_level3_map[level1_id, level2_id, level3_id] = [level1_id_label, level2_id_label, level3_id_label]
                scannet_id = int(line[4])           # original semantic class id
                scannet_id_label = line[7]          # original semantic class label
                mapping[scannet_id] = [level1_id, level2_id, level3_id]
                mapping_label[scannet_id_label] = [level1_id_label, level2_id_label, level3_id_label]
                tree_original_map[scannet_id] = scannet_id_label

            # sort
            sorted_tree_original_map = dict(sorted(tree_original_map.items()))
            sorted_tree_level1_map = dict(sorted(tree_level1_map.items()))
            sorted_tree_level2_map = dict(sorted(tree_level2_map.items()))
            sorted_tree_level3_map = dict(sorted(tree_level3_map.items()))
            tree_id_classes_map = [sorted_tree_level1_map, sorted_tree_level2_map, sorted_tree_level3_map, sorted_tree_original_map]

        return mapping, mapping_label, tree_id_classes_map

    def load_scannet_tree_mapping_nyu40level4(self, path, read_name):
        """ Returns a dict mapping scannet Ids to NYU40 Ids

        Args:
            path: Path to the original scannet data. 
                This is used to get scannetv2-labels.combined.tsv

        Returns:
            mapping: A dict from ints to ints
                example:
                    {1: 1,
                    2: 5,
                    22: 23}

        """
        mapping = {}
        mapping_label = {}
        
        with open(os.path.join(os.path.dirname(path), read_name)) as tsvfile:
            tsvreader = csv.reader(tsvfile, delimiter='\t')
            tree_level1_map = {}
            tree_level2_map = {}
            tree_level3_map = {}
            tree_level4_map = {}
            tree_original_map = {}
            for i, line in enumerate(tsvreader):
                if i==0:
                    continue
                if 17>len(line)-1 or line[17]=="":
                    # level1: 17/18
                    level1_id = None
                    level1_id_label = None
                else:
                    level1_id = int(line[17])
                    level1_id_label = line[18]
                    tree_level1_map[level1_id] = level1_id_label
                if 19>len(line)-1 or line[19]=="":
                    # level2: 19/20
                    level2_id = None
                    level2_id_label = None
                else:
                    level2_id = int(line[19])
                    level2_id_label = line[20]
                    tree_level2_map[level1_id, level2_id] = [level1_id_label, level2_id_label]
                if 21>len(line)-1 or line[21]=="":
                    # level3: 21/22
                    level3_id = None
                    level3_id_label = None
                else:
                    level3_id = int(line[21])
                    level3_id_label = line[22]
                    tree_level3_map[level1_id, level2_id, level3_id] = [level1_id_label, level2_id_label, level3_id_label]
                if 23>len(line)-1 or line[23]=="":
                    # level4: 23/24
                    level4_id = None
                    level4_id_label = None
                else:
                    level4_id = int(line[23])
                    level4_id_label = line[24]
                    tree_level4_map[level1_id, level2_id, level3_id, level4_id] = [level1_id_label, level2_id_label, level3_id_label, level4_id_label]
                
                scannet_id = int(line[4])           # original semantic class id [nyu40id]
                scannet_id_label = line[7]          # original semantic class label [nyu40class]
                mapping[scannet_id] = [level1_id, level2_id, level3_id, level4_id]
                mapping_label[scannet_id_label] = [level1_id_label, level2_id_label, level3_id_label, level4_id_label]
                tree_original_map[scannet_id] = scannet_id_label

            # sort
            sorted_mapping = dict(sorted(mapping.items(), key=lambda item: item[0]))
            sorted_tree_original_map = dict(sorted(tree_original_map.items()))
            sorted_tree_level1_map = dict(sorted(tree_level1_map.items()))
            sorted_tree_level2_map = dict(sorted(tree_level2_map.items()))
            sorted_tree_level3_map = dict(sorted(tree_level3_map.items()))
            sorted_tree_level4_map = dict(sorted(tree_level4_map.items()))
            tree_id_classes_map = [sorted_tree_level1_map, sorted_tree_level2_map, sorted_tree_level3_map, sorted_tree_level4_map, sorted_tree_original_map]

        return sorted_mapping, mapping_label, tree_id_classes_map

    def load_scannet_tree_mapping_rawtolevel4(self, path, read_name):
        """ Returns a dict mapping scannet Ids to NYU40 Ids

        Args:
            path: Path to the original scannet data. 
                This is used to get scannetv2-labels.combined.tsv

        Returns:
            mapping: A dict from ints to ints
                example:
                    {1: 1,
                    2: 5,
                    22: 23}

        """
        mapping = {}
        mapping_label = {}
        
        with open(os.path.join(os.path.dirname(path), read_name)) as tsvfile:
            tsvreader = csv.reader(tsvfile, delimiter='\t')
            tree_level1_map = {}
            tree_level2_map = {}
            tree_level3_map = {}
            tree_level4_map = {}
            tree_original_map = {}
            for i, line in enumerate(tsvreader):
                if i==0:
                    continue
                if 17>len(line)-1 or line[17]=="":
                    # level1: 17/18
                    level1_id = None
                    level1_id_label = None
                else:
                    level1_id = int(line[17])
                    level1_id_label = line[18]
                    tree_level1_map[level1_id] = level1_id_label
                if 19>len(line)-1 or line[19]=="":
                    # level2: 19/20
                    level2_id = None
                    level2_id_label = None
                else:
                    level2_id = int(line[19])
                    level2_id_label = line[20]
                    tree_level2_map[level1_id, level2_id] = [level1_id_label, level2_id_label]
                if 21>len(line)-1 or line[21]=="":
                    # level3: 21/22
                    level3_id = None
                    level3_id_label = None
                else:
                    level3_id = int(line[21])
                    level3_id_label = line[22]
                    tree_level3_map[level1_id, level2_id, level3_id] = [level1_id_label, level2_id_label, level3_id_label]
                if 23>len(line)-1 or line[23]=="":
                    # level4: 23/24
                    level4_id = None
                    level4_id_label = None
                else:
                    level4_id = int(line[23])
                    level4_id_label = line[24]
                    tree_level4_map[level1_id, level2_id, level3_id, level4_id] = [level1_id_label, level2_id_label, level3_id_label, level4_id_label]
                
                scannet_id = int(line[0])           # original semantic class id [rawid]
                scannet_id_label = line[1]          # original semantic class label [rawclass]
                mapping[scannet_id] = [level1_id, level2_id, level3_id, level4_id]
                mapping_label[scannet_id_label] = [level1_id_label, level2_id_label, level3_id_label, level4_id_label]
                tree_original_map[scannet_id] = scannet_id_label

            # sort
            sorted_mapping = dict(sorted(mapping.items(), key=lambda item: item[0]))
            sorted_tree_original_map = dict(sorted(tree_original_map.items()))
            sorted_tree_level1_map = dict(sorted(tree_level1_map.items()))
            sorted_tree_level2_map = dict(sorted(tree_level2_map.items()))
            sorted_tree_level3_map = dict(sorted(tree_level3_map.items()))
            sorted_tree_level4_map = dict(sorted(tree_level4_map.items()))
            tree_id_classes_map = [sorted_tree_level1_map, sorted_tree_level2_map, sorted_tree_level3_map, sorted_tree_level4_map, sorted_tree_original_map]

        return sorted_mapping, mapping_label, tree_id_classes_map

    def load_scannet_tree_mapping_rawtolevel5(self, path, read_name):
        """ Returns a dict mapping scannet Ids to NYU40 Ids

        Args:
            path: Path to the original scannet data. 
                This is used to get scannetv2-labels.combined.tsv

        Returns:
            mapping: A dict from ints to ints
                example:
                    {1: 1,
                    2: 5,
                    22: 23}

        """
        mapping = {}
        mapping_label = {}
        
        with open(os.path.join(os.path.dirname(path), read_name)) as tsvfile:
            tsvreader = csv.reader(tsvfile, delimiter='\t')
            tree_level1_map = {}
            tree_level2_map = {}
            tree_level3_map = {}
            tree_level4_map = {}
            tree_level5_map = {}
            tree_original_map = {}
            for i, line in enumerate(tsvreader):
                if i==0:
                    continue
                if 17>len(line)-1 or line[17]=="":
                    # level1: 17/18
                    level1_id = None
                    level1_id_label = None
                else:
                    level1_id = int(line[17])
                    level1_id_label = line[18]
                    tree_level1_map[level1_id] = level1_id_label
                if 19>len(line)-1 or line[19]=="":
                    # level2: 19/20
                    level2_id = None
                    level2_id_label = None
                else:
                    level2_id = int(line[19])
                    level2_id_label = line[20]
                    # tree_level2_map[level1_id, level2_id] = level2_id_label
                    tree_level2_map[level1_id, level2_id] = [level1_id_label, level2_id_label]
                if 21>len(line)-1 or line[21]=="":
                    # level3: 21/22
                    level3_id = None
                    level3_id_label = None
                else:
                    level3_id = int(line[21])
                    level3_id_label = line[22]
                    # tree_level3_map[level1_id, level2_id, level3_id] = level3_id_label
                    tree_level3_map[level1_id, level2_id, level3_id] = [level1_id_label, level2_id_label, level3_id_label]
                if 23>len(line)-1 or line[23]=="":
                    # level4: 23/24
                    level4_id = None
                    level4_id_label = None
                else:
                    level4_id = int(line[23])
                    level4_id_label = line[24]
                    # tree_level4_map[level1_id, level2_id, level3_id, level4_id] = level4_id_label
                    tree_level4_map[level1_id, level2_id, level3_id, level4_id] = [level1_id_label, level2_id_label, level3_id_label, level4_id_label]
                if 25>len(line)-1 or line[26]=="":
                    # level5: 25/26
                    level5_id = None
                    level5_id_label = None
                else:
                    level5_id = int(line[25])
                    level5_id_label = line[26]
                    # tree_level5_map[level1_id, level2_id, level3_id, level4_id, level5_id] = level5_id_label
                    tree_level5_map[level1_id, level2_id, level3_id, level4_id, level5_id] = [level1_id_label, level2_id_label, level3_id_label, level4_id_label, level5_id_label]
                
                scannet_id = int(line[0])           # original semantic class id [rawid]
                scannet_id_label = line[1]          # original semantic class label [rawclass]
                mapping[scannet_id] = [level1_id, level2_id, level3_id, level4_id, level5_id]
                mapping_label[scannet_id_label] = [level1_id_label, level2_id_label, level3_id_label, level4_id_label, level5_id_label]
                tree_original_map[scannet_id] = scannet_id_label

            # sort
            sorted_mapping = dict(sorted(mapping.items(), key=lambda item: item[0]))
            sorted_tree_original_map = dict(sorted(tree_original_map.items()))
            sorted_tree_level1_map = dict(sorted(tree_level1_map.items()))
            sorted_tree_level2_map = dict(sorted(tree_level2_map.items()))
            sorted_tree_level3_map = dict(sorted(tree_level3_map.items()))
            sorted_tree_level4_map = dict(sorted(tree_level4_map.items()))
            sorted_tree_level5_map = dict(sorted(tree_level5_map.items()))
            tree_id_classes_map = [sorted_tree_level1_map, sorted_tree_level2_map, sorted_tree_level3_map, sorted_tree_level4_map, sorted_tree_level5_map, sorted_tree_original_map]

            # assert(len(mapping) == len(mapping_label))
        return sorted_mapping, mapping_label, tree_id_classes_map

    def load_scannet_label_mapping(self, path, read_name):
        """ Returns a dict mapping scannet category label strings to scannet Ids

        scene****_**.aggregation.json contains the category labels as strings 
        so this maps the strings to the integer scannet Id

        Args:
            path: Path to the original scannet data.
                This is used to get scannetv2-labels.combined.tsv

        Returns:
            mapping: A dict from strings to ints
                example:
                    {'wall': 1,
                    'chair: 2,
                    'books': 22}

        """

        mapping = {}
        mapping_class = {}
        class_str = []
        with open(os.path.join(os.path.dirname(path), read_name)) as tsvfile:
            tsvreader = csv.reader(tsvfile, delimiter='\t')
            for i, line in enumerate(tsvreader):
                if i==0:
                    continue
                scannet_raw_id, scannet_raw_name = int(line[0]), line[2]
                scannet_raw_new_id, scannet_raw_new_name = int(line[27]), line[28]
                mapping[scannet_raw_id] = scannet_raw_new_id
                mapping_class[scannet_raw_new_id] = scannet_raw_new_name
        
        for key, item in mapping_class.items():
            class_str.append(item)

        return mapping, mapping_class, class_str

    def _preprocess_semantic(self, label: np.ndarray, instance: np.ndarray):
        r"""Preprocesses the label map by resizing to :math:`(H, W, C)`, and (optionally) using channels first :math:`(C, H, W)` representation.

        Args:
            label (np.ndarray): Raw input label image

        Retruns:
            np.ndarray: Preprocessed label map

        Shape:
            - Input: :math:`(H_\text{old}, W_\text{old}, C)`
            - Output: :math:`(H, W, C)` if `self.channels_first == False`, else :math:`(C, H, W)`.
        """
        label = cv2.resize(label, (self.desired_width, self.desired_height), interpolation=cv2.INTER_NEAREST)
        instance = cv2.resize(instance, (self.desired_width, self.desired_height), interpolation=cv2.INTER_NEAREST)
        label = np.expand_dims(label, -1)
        instance = np.expand_dims(instance, -1)

        if self.channels_first:
            label = datautils.channels_first(label)
            instance = datautils.channels_first(instance)

        return label, instance
    
    def _preprocess_semantic_label(self, label: np.ndarray):
        r"""Preprocesses the label map by resizing to :math:`(H, W, C)`, and (optionally) using channels first :math:`(C, H, W)` representation.

        Args:
            label (np.ndarray): Raw input label image

        Retruns:
            np.ndarray: Preprocessed label map

        Shape:
            - Input: :math:`(H_\text{old}, W_\text{old}, C)`
            - Output: :math:`(H, W, C)` if `self.channels_first == False`, else :math:`(C, H, W)`.
        """
        # color: INTER_LINEAR, depth: INTER_NEAREST, semantic: INTER_NEAREST
        label = cv2.resize(label, (self.desired_width, self.desired_height), interpolation=cv2.INTER_NEAREST)
        label = np.expand_dims(label, -1)

        if self.channels_first:
            label = datautils.channels_first(label)

        return label


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

def to_scalar(inp: Union[np.ndarray, torch.Tensor, float]) -> Union[int, float]:
    """
    Convert the input to a scalar
    """
    if isinstance(inp, float):
        return inp

    if isinstance(inp, np.ndarray):
        assert inp.size == 1
        return inp.item()

    if isinstance(inp, torch.Tensor):
        assert inp.numel() == 1
        return inp.item()

def as_intrinsics_matrix(intrinsics):
    """
    Get matrix representation of intrinsics.

    """
    K = np.eye(3)
    K[0, 0] = intrinsics[0]
    K[1, 1] = intrinsics[1]
    K[0, 2] = intrinsics[2]
    K[1, 2] = intrinsics[3]
    return K

def readEXR_onlydepth(filename):
    """
    Read depth data from EXR image file.

    Args:
        filename (str): File path.

    Returns:
        Y (numpy.array): Depth buffer in float32 format.
    """
    # move the import here since only CoFusion needs these package
    # sometimes installation of openexr is hard, you can run all other datasets
    # even without openexr
    import Imath
    import OpenEXR as exr

    exrfile = exr.InputFile(filename)
    header = exrfile.header()
    dw = header["dataWindow"]
    isize = (dw.max.y - dw.min.y + 1, dw.max.x - dw.min.x + 1)

    channelData = dict()

    for c in header["channels"]:
        C = exrfile.channel(c, Imath.PixelType(Imath.PixelType.FLOAT))
        C = np.fromstring(C, dtype=np.float32)
        C = np.reshape(C, isize)

        channelData[c] = C

    Y = None if "Y" not in header["channels"] else channelData["Y"]

    return Y

def plot_semantic_legend(
    label, 
    label_name, 
    colormap=None, 
    font_size=30,
    font_path=None,
    save_path=None,
    img_name=None,
    save_name = "semantic_class_Legend"):


    """Plot Colour Legend for Semantic Classes

    Parameters
    ----------
    label: numpy.ndarray, (N,), int
        One-dimensional array containing the unique labels of exsiting semantic classes
    label_names: list of string
        Label id to label name.
    font_size: int
        Font size (default: 30).
    colormap: numpy.ndarray, (M, 3), numpy.uint8
        Label id to color.
        By default, :func:`~imgviz.label_colormap` is used.
    font_path: str
        Font path.

    Returns
    -------
    res: numpy.ndarray, (H, W, 3), numpy.uint8
    Legend image of visualising semantic labels.

    """

    label = np.unique(label)
    if colormap is None:
        colormap = label_colormap()

    text_sizes = np.array(
            [
                draw_module.text_size(
                    label_name[l], font_size, font_path=font_path
                )
                for l in label
            ]
        )

    text_height, text_width = text_sizes.max(axis=0)
    legend_height = text_height * len(label) + 5
    legend_width = text_width + 20 + (text_height - 10)


    legend = np.zeros((legend_height+50, legend_width+50, 3), dtype=np.uint8)
    aabb1 = np.array([25, 25], dtype=float)
    aabb2 = aabb1 + (legend_height, legend_width)

    legend = draw_module.rectangle(
        legend, aabb1, aabb2, fill=(255, 255, 255)
    )  # fill the legend area by white colour

    y1, x1 = aabb1.round().astype(int)
    y2, x2 = aabb2.round().astype(int)

    for i, l in enumerate(label):
        box_aabb1 = aabb1 + (i * text_height + 5, 5)
        box_aabb2 = box_aabb1 + (text_height - 10, text_height - 10)
        legend = draw_module.rectangle(
            legend, aabb1=box_aabb1, aabb2=box_aabb2, fill=colormap[l]
        )
        legend = draw_module.text(
            legend,
            yx=aabb1 + (i * text_height, 10 + (text_height - 10)),
            text=label_name[l],
            size=font_size,
            font_path=font_path,
            )

    
    plt.figure(1)
    plt.title("Semantic Legend!")
    plt.imshow(legend)
    plt.axis("off")

    img_arr = imgviz.io.pyplot_to_numpy()
    plt.close()
    if save_path is not None:
        if img_name is not None:
            sav_dir = os.path.join(save_path, img_name)
        else:
            sav_dir = os.path.join(save_path, save_name+".png")
        # plt.savefig(sav_dir, bbox_inches='tight', pad_inches=0)
        cv2.imwrite(sav_dir, img_arr[:,:,::-1])
    return img_arr

def find_max_level3(label_mapping_tree):
    level_items = []
    for key, value in label_mapping_tree.items():
        level_items.append(value)
    level_items = np.asarray(level_items)
    return np.max(level_items[:,0]), np.max(level_items[:,1]), np.max(level_items[:,2])

def find_max_level4(label_mapping_tree):
    level_items = []
    for key, value in label_mapping_tree.items():
        level_items.append(value)
    level_items = np.asarray(level_items)
    return np.max(level_items[:,0]), np.max(level_items[:,1]), np.max(level_items[:,2]), np.max(level_items[:,3])

def find_max_level5(label_mapping_tree):
    level_items = []
    for key, value in label_mapping_tree.items():
        level_items.append(value)
    level_items = np.asarray(level_items)
    return np.max(level_items[:,0]), np.max(level_items[:,1]), np.max(level_items[:,2]), np.max(level_items[:,3]), np.max(level_items[:,4])

def find_max_level(label_mapping_tree, flag_add=False):
    max_level_list = []
    level_items = []
    
    for key, value in label_mapping_tree.items():
        level_items.append(value)
    level_items = np.asarray(level_items)

    for i_level in range(level_items.shape[1]):
        if flag_add:
            max_level_list.append(np.max(level_items[:,i_level])+1)
        else:
            max_level_list.append(np.max(level_items[:,i_level]))
        
    return max_level_list

def visual_semantic_legend(classes, classes_name, colormap_use, save_path_use, save_name_use):

    iter = 40
    iter_num = 0
    for i in tqdm(range(0, len(classes), iter)):
        i_begin = i
        i_end = i_begin + iter -1
        if i_end > len(classes)-1:
            i_end = len(classes)-1
        
        legend_tree = plot_semantic_legend(np.arange(0,i_end-i_begin), classes_name[i_begin:i_end], 
            colormap = colormap_use[i_begin:i_end, :], save_path=save_path_use, save_name = save_name_use+str(iter_num))
        
        iter_num += 1

    for i_img in range(iter_num):
        img = cv2.imread(os.path.join(save_path_use, save_name_use+str(i_img)+".png"))
        if i_img == 0:
            img_concat = img
        else:
            if img_concat.shape[0] == img.shape[0] and img_concat.shape[2] == img.shape[2]:
                img_concat = np.concatenate((img_concat, img), axis=1)
            else:
                if img_concat.shape[0]>img.shape[0]:
                    padding = np.full((img_concat.shape[0]-img.shape[0], img.shape[1], 3), 255)
                    image_padded = np.concatenate((img, padding), axis=0)
                    img_concat = np.concatenate((img_concat, image_padded), axis=1)
                elif img_concat.shape[0]<img.shape[0]:
                    padding = np.full((img.shape[0] - img_concat.shape[0], img_concat.shape[1], 3), 255)
                    img_concat_padded = np.concatenate((img_concat, padding), axis=0)
                    img_concat = np.concatenate((img_concat_padded, img), axis=1)
                pass
            
    cv2.imwrite(os.path.join(save_path_use, save_name_use+".png"), img_concat)
