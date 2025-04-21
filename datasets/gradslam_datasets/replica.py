import glob
import os
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from natsort import natsorted
import imageio
import cv2, tqdm
import json
from imgviz import label_colormap
import imageio
import cv2
import imgviz
from imgviz import label_colormap
from imgviz import draw as draw_module
import matplotlib.pyplot as plt
import yaml
from tqdm import tqdm
import time


from .basedataset import GradSLAMDataset, readEXR_onlydepth, as_intrinsics_matrix
# from datasets.gradslam_datasets import plot_semantic_legend
from . import datautils
import imageio


class ReplicaDataset(GradSLAMDataset):
    def __init__(
        self,
        config_dict,
        basedir,
        sequence,
        stride: Optional[int] = None,
        start: Optional[int] = 0,
        end: Optional[int] = -1,
        desired_height: Optional[int] = 480,
        desired_width: Optional[int] = 640,
        load_embeddings: Optional[bool] = False,
        embedding_dir: Optional[str] = "embeddings",
        embedding_dim: Optional[int] = 512,
        **kwargs,
    ):
        self.input_folder = os.path.join(basedir, sequence)
        self.pose_path = os.path.join(self.input_folder, "traj.txt")
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
        color_paths = natsorted(glob.glob(f"{self.input_folder}/results/frame*.jpg"))
        depth_paths = natsorted(glob.glob(f"{self.input_folder}/results/depth*.png"))
        embedding_paths = None
        if self.load_embeddings:
            embedding_paths = natsorted(glob.glob(f"{self.input_folder}/{self.embedding_dir}/*.pt"))
        return color_paths, depth_paths, embedding_paths

    def load_poses(self):
        poses = []
        with open(self.pose_path, "r") as f:
            lines = f.readlines()
        for i in range(self.num_imgs):
            line = lines[i]
            c2w = np.array(list(map(float, line.split()))).reshape(4, 4)
            # c2w[:3, 1] *= -1
            # c2w[:3, 2] *= -1
            c2w = torch.from_numpy(c2w).float()
            poses.append(c2w)
        return poses

    def read_embedding_from_file(self, embedding_file_path):
        embedding = torch.load(embedding_file_path)
        return embedding.permute(0, 2, 3, 1)  # (1, H, W, embedding_dim)
    
class ReplicaDataset_semantic(GradSLAMDataset):
    def __init__(
        self,
        config_dict,
        basedir,
        sequence,
        stride: Optional[int] = None,
        start: Optional[int] = 0,
        end: Optional[int] = -1,
        desired_height: Optional[int] = 480,
        desired_width: Optional[int] = 640,
        load_embeddings: Optional[bool] = False,
        embedding_dir: Optional[str] = "embeddings",
        embedding_dim: Optional[int] = 512,
        **kwargs,
    ):
        # read setting ======================
        self.flag_load_semantic = False
        flag_save = True
        # to save time, set [flag_save = True & self.flag_load_semantic = False] for the first time running. 
        # after that, set [flag_save = False & self.flag_load_semantic = True] to read semantic hierarchy directly from saved path.

        # for original, no need to save/load hierarchy
        if config_dict["sem_mode"] == "original":
            self.flag_load_semantic = False
            flag_save = False

        # setting
        self.sem_mode = config_dict["sem_mode"]
        self.results_dir = config_dict["results_dir"]
        self.use_pyramid = config_dict["use_pyramid"]
        self.pyramid_level = config_dict["pyramid_level"]  
        self.num_tree_level = config_dict["num_tree_level"]
        self.dataset_name = config_dict["dataset_name"]

        self.input_folder = os.path.join(basedir, sequence)
        self.pose_path = os.path.join(self.input_folder, "traj.txt")
        self.input_folder_sem = os.path.join(config_dict["basedir_sem"], sequence)
        self.semantic_paths = natsorted(glob.glob(f"{self.input_folder_sem}/semantic_class/semantic_class_*.png"))
        if start<end and start>=0:
            self.semantic_paths = self.semantic_paths[start : end : stride]

        # 1. semantic label info read
        if self.sem_mode == "original":
            with open(os.path.join(self.input_folder_sem, "info_semantic.json"), "r") as f:
                annotations = json.load(f)
            instance_id_to_semantic_label_id = np.array(annotations["id_to_label"])
            label_colour_map = label_colormap()
            self.colour_map_np = label_colour_map
            self.colors_map_all = label_colour_map
            self.semantic_class = read_semantic_classes(annotations)
            self.num_semantic = len(self.semantic_class)
            self.num_semantic_class = self.num_semantic
        elif "tree" in self.sem_mode:
            name_to_read = "info_semantic_tree.json"

            with open(os.path.join(self.input_folder_sem, name_to_read), "r") as f:
                annotations = json.load(f)
            label_mapping_tree, tree_id_classes_map = read_tree_annotation(annotations, tree_level=self.num_tree_level)
            num_level_list = find_max_level(label_mapping_tree, flag_add=True)
            num_level_list.append(len(label_mapping_tree))
            self.num_semantic = num_level_list
            self.label_mapping_tree = label_mapping_tree
            self.tree_id_classes_map = tree_id_classes_map

            # get leaf info
            tree_id_classes_map_leaf = tree_id_classes_map[-2]
            self.colors_map_all = imgviz.label_colormap(self.num_semantic[-1])
            class_name_original = [ (str(key)+": "+str(value)) for key, value in tree_id_classes_map[-1].items() ]
            self.colour_map_np_level = {}
            self.num_semantic_class = self.colors_map_all.shape[0]
            classes_name = []
            for idx, (key, value) in enumerate(label_mapping_tree.items(), 0):
                value = torch.tensor(value)
                self.colour_map_np_level[value] = self.colors_map_all[idx, :]
                label_name = tree_id_classes_map[-1][int(key)]
                label_name_string = np.array2string(value.numpy()) + "_" + str(idx) + "_" + label_name  # Convert tensor value to string
                classes_name.append(label_name_string)

            vis_label_save_dir = os.path.join(self.input_folder_sem, "vis_semantic_label")
            if not os.path.exists(vis_label_save_dir):
                os.makedirs(vis_label_save_dir, exist_ok=True)

            # visual color bar (optional)
            legend_tree = plot_semantic_legend(np.arange(0,38), classes_name[:38], 
                        colormap =  self.colors_map_all[:38], save_path=self.results_dir, save_name="semantic_class_Legend_leaf1")
            legend_tree = plot_semantic_legend(np.arange(0,38), classes_name[38:76], 
                        colormap =  self.colors_map_all[38:76], save_path=self.results_dir, save_name="semantic_class_Legend_leaf2")
            legend_tree = plot_semantic_legend(np.arange(0,26), classes_name[76:], 
                        colormap =  self.colors_map_all[76:], save_path=self.results_dir, save_name="semantic_class_Legend_leaf3")
            
            img1 = cv2.imread(os.path.join(self.results_dir, "semantic_class_Legend_leaf1.png"))
            img2 = cv2.imread(os.path.join(self.results_dir, "semantic_class_Legend_leaf2.png"))
            img3 = cv2.imread(os.path.join(self.results_dir, "semantic_class_Legend_leaf3.png"))
            img_concat = np.concatenate((img1, img2, img3), axis=1)
            cv2.imwrite(os.path.join(self.results_dir, "semantic_class_Legend_leaf.png"), img_concat)

            legend_tree = plot_semantic_legend(np.arange(0,38), class_name_original[:38], 
                        font_size = 30, colormap =  self.colors_map_all[:38], save_path=self.results_dir, save_name="semantic_class_Legend_original1")
            legend_tree = plot_semantic_legend(np.arange(0,38), class_name_original[38:76], 
                        font_size = 30, colormap =  self.colors_map_all[38:76], save_path=self.results_dir, save_name="semantic_class_Legend_original2") 
            legend_tree = plot_semantic_legend(np.arange(0,26), class_name_original[76:], 
                        font_size = 30, colormap =  self.colors_map_all[76:], save_path=self.results_dir, save_name="semantic_class_Legend_original3")  

            img1 = cv2.imread(os.path.join(self.results_dir, "semantic_class_Legend_original1.png"))
            img2 = cv2.imread(os.path.join(self.results_dir, "semantic_class_Legend_original2.png"))
            img3 = cv2.imread(os.path.join(self.results_dir, "semantic_class_Legend_original3.png"))
            img_concat = np.concatenate((img1, img2, img3), axis=1)
            cv2.imwrite(os.path.join(self.results_dir, "semantic_class_Legend_original.png"), img_concat)      

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
        assert(len(self.semantic_paths)==len(self.color_paths))
        assert(len(self.semantic_paths)==self.num_imgs)

        # 2. batch processing for semantic read info
        if not self.flag_load_semantic:
            if self.sem_mode == "original":
                pass
            elif "tree" in self.sem_mode:
                if self.sem_mode == "tree":
                    # save_filename = "semantic_labels_tree5level"
                    save_filename = "semantic_labels_tree5level_test"
                
                # load semantic original info
                semantic_original = []
                print("processing {} frames. ******************** ".format(self.num_imgs))
                for item in tqdm(range(self.num_imgs)):
                    # load semantic raw info
                    label = np.asarray(imageio.imread(self.semantic_paths[item]), dtype=np.int64)
                    # resize & add list
                    label = self._preprocess_semantic_label(label)
                    semantic_original.append(label)

                semantic_original = np.asarray(semantic_original) # [num_img, h, w， 1]

                # transfer to tree-level =================================================
                semantic_raw_tree_all = []
                if self.num_tree_level == 5:
                    semantic_raw_tree_level0 = semantic_original.copy()
                    semantic_raw_tree_level1 = semantic_original.copy()
                    semantic_raw_tree_level2 = semantic_original.copy()
                    semantic_raw_tree_level3 = semantic_original.copy()
                    semantic_raw_tree_level4 = semantic_original.copy()

                    for scan_id, tree_id in tqdm(label_mapping_tree.items(), desc="Processing Semantic Labels to Hierarchy"):
                        scan_id = int(scan_id)
                        semantic_raw_tree_level0[semantic_original==scan_id] = tree_id[0]
                        semantic_raw_tree_level1[semantic_original==scan_id] = tree_id[1]
                        semantic_raw_tree_level2[semantic_original==scan_id] = tree_id[2]
                        semantic_raw_tree_level3[semantic_original==scan_id] = tree_id[3]
                        semantic_raw_tree_level4[semantic_original==scan_id] = tree_id[4]
                    
                    semantic_raw_tree_all.append(semantic_raw_tree_level0.squeeze())
                    semantic_raw_tree_all.append(semantic_raw_tree_level1.squeeze())
                    semantic_raw_tree_all.append(semantic_raw_tree_level2.squeeze())
                    semantic_raw_tree_all.append(semantic_raw_tree_level3.squeeze())
                    semantic_raw_tree_all.append(semantic_raw_tree_level4.squeeze())
                    semantic_raw_tree_all.append(semantic_original.squeeze())          # original
                elif self.num_tree_level == 4:
                    semantic_raw_tree_level0 = semantic_original.copy()
                    semantic_raw_tree_level1 = semantic_original.copy()
                    semantic_raw_tree_level2 = semantic_original.copy()
                    semantic_raw_tree_level3 = semantic_original.copy()

                    for scan_id, tree_id in tqdm(label_mapping_tree.items(), desc="Processing Semantic Labels to Hierarchy"):
                        scan_id = int(scan_id)
                        semantic_raw_tree_level0[semantic_original==scan_id] = tree_id[0]
                        semantic_raw_tree_level1[semantic_original==scan_id] = tree_id[1]
                        semantic_raw_tree_level2[semantic_original==scan_id] = tree_id[2]
                        semantic_raw_tree_level3[semantic_original==scan_id] = tree_id[3]
                    
                    semantic_raw_tree_all.append(semantic_raw_tree_level0.squeeze())
                    semantic_raw_tree_all.append(semantic_raw_tree_level1.squeeze())
                    semantic_raw_tree_all.append(semantic_raw_tree_level2.squeeze())
                    semantic_raw_tree_all.append(semantic_raw_tree_level3.squeeze())
                    semantic_raw_tree_all.append(semantic_original.squeeze())          # original
                elif self.num_tree_level == 3:
                    semantic_raw_tree_level0 = semantic_original.copy()
                    semantic_raw_tree_level1 = semantic_original.copy()
                    semantic_raw_tree_level2 = semantic_original.copy()

                    for scan_id, tree_id in tqdm(label_mapping_tree.items(), desc="Processing Semantic Labels to Hierarchy"):
                        scan_id = int(scan_id)
                        semantic_raw_tree_level0[semantic_original==scan_id] = tree_id[0]
                        semantic_raw_tree_level1[semantic_original==scan_id] = tree_id[1]
                        semantic_raw_tree_level2[semantic_original==scan_id] = tree_id[2]
                        
                    semantic_raw_tree_all.append(semantic_raw_tree_level0.squeeze())
                    semantic_raw_tree_all.append(semantic_raw_tree_level1.squeeze())
                    semantic_raw_tree_all.append(semantic_raw_tree_level2.squeeze())
                    semantic_raw_tree_all.append(semantic_original.squeeze())          # original
                elif self.num_tree_level == 2:
                    semantic_raw_tree_level0 = semantic_original.copy()
                    semantic_raw_tree_level1 = semantic_original.copy()

                    for scan_id, tree_id in tqdm(label_mapping_tree.items(), desc="Processing Semantic Labels to Hierarchy"):
                        scan_id = int(scan_id)
                        semantic_raw_tree_level0[semantic_original==scan_id] = tree_id[0]
                        semantic_raw_tree_level1[semantic_original==scan_id] = tree_id[1]
                    
                    semantic_raw_tree_all.append(semantic_raw_tree_level0.squeeze())
                    semantic_raw_tree_all.append(semantic_raw_tree_level1.squeeze())
                    semantic_raw_tree_all.append(semantic_original.squeeze())          # original
                
                
                semantic_raw_tree_all = np.asarray(semantic_raw_tree_all)           
                self.semantic_labels_raw_gt =semantic_raw_tree_all
                # =======================================================================

            # save transfered semantic label
            if flag_save:
                sem_file_save_path = os.path.join(self.input_folder_sem, save_filename)
                print("semantic label save path is => ", sem_file_save_path)
                os.makedirs(sem_file_save_path, exist_ok=True)
                np.save(os.path.join(sem_file_save_path, "semantic_labels_raw_gt.npy"), self.semantic_labels_raw_gt)
                for i in tqdm(range(0, self.num_imgs), desc="Saving"):
                    save_name = os.path.splitext(os.path.basename(self.semantic_paths[i]))[0]
                    np.save(os.path.join(sem_file_save_path, "{}.npy".format(save_name)), self.semantic_labels_raw_gt[:, i, :, :])

        # if load info, change paths
        if self.flag_load_semantic:
            if self.sem_mode=="original":
                self.load_sem_path = os.path.join(self.input_folder_sem, "semantic_labels_original")
                for i in range(len(self.semantic_paths)):
                    self.semantic_paths[i] = self.semantic_paths[i].replace("semantic_class", "semantic_labels_original").replace("png", "npy")
            if "tree" in self.sem_mode:
                if self.sem_mode == "tree":
                    save_filename = "semantic_labels_tree5level"

                self.load_sem_path = os.path.join(self.input_folder_sem, save_filename)
                print("semantic label load path is => ", self.load_sem_path)
                for i in range(len(self.semantic_paths)):
                    self.semantic_paths[i] = self.semantic_paths[i].replace("semantic_class", save_filename, 1).replace("png", "npy")

    def get_filepaths(self):
        color_paths = natsorted(glob.glob(f"{self.input_folder}/results/frame*.jpg"))
        depth_paths = natsorted(glob.glob(f"{self.input_folder}/results/depth*.png"))
        embedding_paths = None
        if self.load_embeddings:
            embedding_paths = natsorted(glob.glob(f"{self.input_folder}/{self.embedding_dir}/*.pt"))
        return color_paths, depth_paths, embedding_paths

    def load_poses(self):
        poses = []
        with open(self.pose_path, "r") as f:
            lines = f.readlines()
        for i in range(self.num_imgs):
            line = lines[i]
            c2w = np.array(list(map(float, line.split()))).reshape(4, 4)
            # c2w[:3, 1] *= -1
            # c2w[:3, 2] *= -1
            c2w = torch.from_numpy(c2w).float()
            poses.append(c2w)
        return poses

    def read_embedding_from_file(self, embedding_file_path):
        embedding = torch.load(embedding_file_path)
        return embedding.permute(0, 2, 3, 1)  # (1, H, W, embedding_dim)
    
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
        label = cv2.resize(label, (self.desired_width, self.desired_height), interpolation=cv2.INTER_NEAREST)
        label = np.expand_dims(label, -1)

        if self.channels_first:
            label = datautils.channels_first(label)

        return label

    def __getitem__(self, index):
        color_path = self.color_paths[index]
        depth_path = self.depth_paths[index]
        # color
        color = np.asarray(imageio.imread(color_path), dtype=float)
        color = self._preprocess_color(color)
        # depth
        if ".png" in depth_path:
            depth = np.asarray(imageio.imread(depth_path), dtype=np.int64)
        elif ".exr" in depth_path:
            depth = readEXR_onlydepth(depth_path)
        # semantic
        if self.flag_load_semantic:
            semantic = np.load(self.semantic_paths[index])
        elif not self.flag_load_semantic and self.sem_mode == "original":
            semantic  = np.asarray(imageio.imread(self.semantic_paths[index]), dtype=float)     # add sem
        elif not self.flag_load_semantic and "tree" in self.sem_mode:
            semantic = self.semantic_labels_raw_gt[:, index, :, :]         # [level_num, num_img, h, w]
            semantic = semantic.squeeze()

        K = as_intrinsics_matrix([self.fx, self.fy, self.cx, self.cy])
        if self.distortion is not None:
            # undistortion is only applied on color image, not depth!
            color = cv2.undistort(color, K, self.distortion)

        color = torch.from_numpy(color)
        K = torch.from_numpy(K)
        semantic = torch.from_numpy(semantic)
        if self.sem_mode == "original":
            semantic = semantic.unsqueeze(2)

        depth = self._preprocess_depth(depth)
        depth = torch.from_numpy(depth)

        K = datautils.scale_intrinsics(K, self.height_downsample_ratio, self.width_downsample_ratio)
        intrinsics = torch.eye(4).to(K)
        intrinsics[:3, :3] = K

        pose = self.transformed_poses[index]

        if self.load_embeddings:
            embedding = self.read_embedding_from_file(self.embedding_paths[index])
            return (
                color.to(self.device).type(self.dtype),
                depth.to(self.device).type(self.dtype),
                intrinsics.to(self.device).type(self.dtype),
                pose.to(self.device).type(self.dtype),
                semantic.to(self.device),                   # add sem
                embedding.to(self.device),  # Allow embedding to be another dtype
                # self.retained_inds[index].item(),
            )

        return (
            color.to(self.device).type(self.dtype),
            depth.to(self.device).type(self.dtype),
            intrinsics.to(self.device).type(self.dtype),
            pose.to(self.device).type(self.dtype),
            semantic.to(self.device),                       # add sem
            # self.retained_inds[index].item(),
        )
    

class ReplicaV2Dataset(GradSLAMDataset):
    def __init__(
        self,
        config_dict,
        basedir,
        sequence,
        use_train_split: Optional[bool] = True,
        stride: Optional[int] = None,
        start: Optional[int] = 0,
        end: Optional[int] = -1,
        desired_height: Optional[int] = 480,
        desired_width: Optional[int] = 640,
        load_embeddings: Optional[bool] = False,
        embedding_dir: Optional[str] = "embeddings",
        embedding_dim: Optional[int] = 512,
        **kwargs,
    ):
        self.use_train_split = use_train_split
        if self.use_train_split:
            self.input_folder = os.path.join(basedir, sequence, "imap/00")
            self.pose_path = os.path.join(self.input_folder, "traj_w_c.txt")
        else:
            self.train_input_folder = os.path.join(basedir, sequence, "imap/00")
            self.train_pose_path = os.path.join(self.train_input_folder, "traj_w_c.txt")
            self.input_folder = os.path.join(basedir, sequence, "imap/01")
            self.pose_path = os.path.join(self.input_folder, "traj_w_c.txt")
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
        if self.use_train_split:
            color_paths = natsorted(glob.glob(f"{self.input_folder}/rgb/rgb_*.png"))
            depth_paths = natsorted(glob.glob(f"{self.input_folder}/depth/depth_*.png"))
        else:
            first_train_color_path = f"{self.train_input_folder}/rgb/rgb_0.png"
            first_train_depth_path = f"{self.train_input_folder}/depth/depth_0.png"
            color_paths = [first_train_color_path] + natsorted(glob.glob(f"{self.input_folder}/rgb/rgb_*.png"))
            depth_paths = [first_train_depth_path] + natsorted(glob.glob(f"{self.input_folder}/depth/depth_*.png"))
        embedding_paths = None
        if self.load_embeddings:
            embedding_paths = natsorted(glob.glob(f"{self.input_folder}/{self.embedding_dir}/*.pt"))
        return color_paths, depth_paths, embedding_paths

    def load_poses(self):
        poses = []
        if not self.use_train_split:
            with open(self.train_pose_path, "r") as f:
                train_lines = f.readlines()
            first_train_frame_line = train_lines[0]
            first_train_frame_c2w = np.array(list(map(float, first_train_frame_line.split()))).reshape(4, 4)
            first_train_frame_c2w = torch.from_numpy(first_train_frame_c2w).float()
            poses.append(first_train_frame_c2w)
        with open(self.pose_path, "r") as f:
            lines = f.readlines()
        if self.use_train_split:
            num_poses = self.num_imgs
        else:
            num_poses = self.num_imgs - 1
        for i in range(num_poses):
            line = lines[i]
            c2w = np.array(list(map(float, line.split()))).reshape(4, 4)
            # c2w[:3, 1] *= -1
            # c2w[:3, 2] *= -1
            c2w = torch.from_numpy(c2w).float()
            poses.append(c2w)
        return poses

    def read_embedding_from_file(self, embedding_file_path):
        embedding = torch.load(embedding_file_path)
        return embedding.permute(0, 2, 3, 1)  # (1, H, W, embedding_dim)
    

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

def read_semantic_classes(annotations):
    """
    read json and transfer to semantic classes
    add void class in id=0
    """
    dataset_classes_raw = {}
    dataset_classes = []
    for item in annotations["classes"]:
        # print(item["classes"], item["id"], item["parents"])
        dataset_classes_raw[item["id"]] = item["name"]
    dataset_classes_raw[0] = "void"
    dataset_classes_raw = dict(sorted(dataset_classes_raw.items(), key=lambda item: item[0]))
    for key, value in dataset_classes_raw.items():
        dataset_classes.append(value)
    return dataset_classes

def read_tree_annotation(annotations, tree_level = 5):
    """
    return: mapping_id_dict, semantic_info_level
    mapping_id_dict/semantic_info_level: sorted by base-id
    mapping_id_dict: sorted by base-id, each base-if corespond to multi-id
    semantic_info_level: sorted by level, last level is base-id, id <-> name
    """

    # for each semantic label
    mapping_id_dict = {}
    # for each level
    semantic_info_level = []
    for i_init in range(tree_level):
        dict_init = {}
        semantic_info_level.append(dict_init)
    
    dict_base = {} 
    for idx, (key, item) in enumerate(annotations.items()):     # each class
        keywords = key.split("_")
        base_id = keywords[0]
        base_name = keywords[1]
        dict_base[int(base_id)] = base_name

        semantic_id_level = []
        semantic_id_level = [-1 for _ in range(tree_level)]  

        for i_level in range(len(item)):    # each level
            level_info = item[i_level]
            for key, value in level_info.items():
                # key:id; value:id_name
                id_level = int(key)
                
                semantic_id_level[i_level] = id_level
                # print("-> in level {}, id and name are: {}, {}".format(i_level, id_level, name_level))
            
        # for semantic_info_level
        for i_level in range(len(item)):    # each level
            level_info = item[i_level]
            for key, value in level_info.items(): 
                semantic_info_level[i_level][tuple(semantic_id_level[:i_level+1])] = value

        mapping_id_dict[base_id] = tuple(semantic_id_level)

    semantic_info_level.append(dict_base)
    return mapping_id_dict, semantic_info_level


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
    
    