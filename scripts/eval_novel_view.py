import argparse
import os
import random
import sys
import shutil
from importlib.machinery import SourceFileLoader
import re, time

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.insert(0, _BASE_DIR)

print("System Paths:")
for p in sys.path:
    print(p)

import matplotlib.pyplot as plt
import cv2
import numpy as np
import torch
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
)
from utils.common_utils import seed_everything
from utils.eval_helpers import eval_newrender, eval_nvs, eval_semantic_newrender, eval_semantic_tree_newrender, show_semantic


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


def load_scene_data(scene_path):
    path = os.path.join(scene_path, 'params.npz')
    print("param load path is: ", path)
    params = dict(np.load(path, allow_pickle=True))
    params = {k: torch.tensor(params[k]).cuda().float().requires_grad_(True) for k in params.keys()}
    return params


if __name__=="__main__":
    time1 = time.time()

    parser = argparse.ArgumentParser()

    parser.add_argument("experiment", type=str, help="Path to experiment file")

    args = parser.parse_args()

    experiment = SourceFileLoader(
        os.path.basename(args.experiment), args.experiment
    ).load_module()

    config = experiment.config

    # Set Experiment Seed
    seed_everything(seed=experiment.config['seed'])
    device = torch.device(config["primary_device"])
    flag_mlp_use = config["model"]["flag_use_embedding"]

    # Create Results Directory and Copy Config
    results_dir = os.path.join(
        experiment.config["workdir"], experiment.config["run_name"]
    )
    experiment.config["data"]["results_dir"] = results_dir

    time2 = time.time()

    print("1. load experiments: ", time2-time1, "s")

    # Load Dataset
    print("Loading Dataset ...")
    dataset_config = config["data"]
    if "gradslam_data_cfg" not in dataset_config:
        gradslam_data_cfg = {}
        gradslam_data_cfg["dataset_name"] = dataset_config["dataset_name"]
    else:
        gradslam_data_cfg = load_dataset_config(dataset_config["gradslam_data_cfg"])
        gradslam_data_cfg = {**gradslam_data_cfg, **dataset_config}
        # gradslam_data_cfg["dataset_name"] = "scannet"
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
        device=device,
        relative_pose=True,
        ignore_bad=dataset_config["ignore_bad"],
        use_train_split=dataset_config["use_train_split"],
    )
    num_frames = dataset_config["num_frames"]
    if num_frames == -1:
        num_frames = len(dataset)

    # semantic
    gradslam_data_cfg["results_dir"] = results_dir
    flag_use_semantic = False
    if "semantic" in gradslam_data_cfg["dataset_name"]:
        print(" ********** USE SEMANTIC ********** ")
        flag_use_semantic = True
        gradslam_data_cfg["sem_mode"] = config["data"]["sem_mode"]
        if "scannet" in gradslam_data_cfg["dataset_name"]:
            num_semantic = dataset.num_semantic
        elif "replica" in gradslam_data_cfg["dataset_name"]:
            num_semantic = dataset.num_semantic
        print("num_semantic is: ", num_semantic)
        if flag_mlp_use == 1:
            num_semantic_dim = sum(dataset.num_semantic[:-1])
            print("num_semantic_dim is: ", num_semantic_dim)

    if flag_mlp_use == 1:
        MLP_func = torch.nn.Conv2d(num_semantic_dim, dataset.num_semantic_class, kernel_size=1)
        MLP_func.load_state_dict(torch.load(results_dir+'/Semantic.pth'))
        MLP_func.cuda()
        MLP_func.eval()

    time3 = time.time()
    print("2. load dataset: (s) (min)", (time3-time2), (time3-time2)/60.0)

    # Load Scene params
    scene_path = results_dir
    params = load_scene_data(scene_path)
    time4 = time.time()
    print("3. load trained params: ", time4-time3, "s")

    if dataset_config['use_train_split']:
        eval_dir = os.path.join(results_dir, "eval_train")
        wandb_name = config['wandb']['name'] + "_Train_Split"
    else:
        eval_dir = os.path.join(results_dir, "eval_nvs")
        wandb_name = config['wandb']['name'] + "_NVS_Split"
    
    # Init WandB
    if config['use_wandb']:
        wandb_time_step = 0
        wandb_tracking_step = 0
        wandb_mapping_step = 0
        wandb_run = wandb.init(project=config['wandb']['project'],
                               entity=config['wandb']['entity'],
                               group=config['wandb']['group'],
                               name=wandb_name,
                               config=config)
        with open(os.path.join(results_dir,'wandb_path_eval.txt'),'w') as f: 
            f.writelines(wandb_run.url) 
            f.writelines("\n")
            f.writelines(wandb_run.dir) 
        
    time5 = time.time()
    print("4. wandb: ", time5-time4, "s")
    

    # Evaluate Final Parameters
    with torch.no_grad():
        if config['use_wandb']:
            if dataset_config['use_train_split']:
                if dataset.sem_mode == "binary1" or dataset.sem_mode == "eigen13" or dataset.sem_mode == "nyu40" or dataset.sem_mode == "original" or dataset.sem_mode == "raw":
                    # original 
                    eval_semantic_newrender(dataset, params, num_frames, eval_dir, sil_thres=config['mapping']['sil_thres'],
                            wandb_run=wandb_run, wandb_save_qual=config['wandb']['eval_save_qual'],
                            mapping_iters=config['mapping']['num_iters'], add_new_gaussians=config['mapping']['add_new_gaussians'],
                            eval_every=config['eval_every'], save_frames=True)
                else:
                    # tree
                    if flag_mlp_use ==1:
                        show_semantic(dataset, params, num_frames, eval_dir, 
                            eval_every=config['eval_every'], save_frames=True, flag_mlp=flag_mlp_use, mlp_func=MLP_func)
                        eval_semantic_tree_newrender(dataset, params, num_frames, eval_dir, sil_thres=config['mapping']['sil_thres'],
                            wandb_run=wandb_run, wandb_save_qual=config['wandb']['eval_save_qual'],
                            mapping_iters=config['mapping']['num_iters'], add_new_gaussians=config['mapping']['add_new_gaussians'],
                            eval_every=config['eval_every'], save_frames=True, flag_mlp=flag_mlp_use, mlp_func=MLP_func, gt_transfer=config['model']['eval_gt_transfer'])
                    else:
                        eval_semantic_tree_newrender(dataset, params, num_frames, eval_dir, sil_thres=config['mapping']['sil_thres'],
                            wandb_run=wandb_run, wandb_save_qual=config['wandb']['eval_save_qual'],
                            mapping_iters=config['mapping']['num_iters'], add_new_gaussians=config['mapping']['add_new_gaussians'],
                            eval_every=config['eval_every'], save_frames=True)
            else:
                eval_nvs(dataset, params, num_frames, eval_dir, sil_thres=config['mapping']['sil_thres'],
                    wandb_run=wandb_run, wandb_save_qual=config['wandb']['eval_save_qual'],
                    mapping_iters=config['mapping']['num_iters'], add_new_gaussians=config['mapping']['add_new_gaussians'],
                    eval_every=config['eval_every'], save_frames=True)
        else:
            if dataset_config['use_train_split']:
                # replica
                if not flag_use_semantic:
                    eval_newrender(dataset, params, num_frames, eval_dir, sil_thres=config['mapping']['sil_thres'],
                                mapping_iters=config['mapping']['num_iters'], add_new_gaussians=config['mapping']['add_new_gaussians'],
                                eval_every=config['eval_every'])
                elif flag_use_semantic:
                    if dataset.sem_mode == "original":
                        eval_semantic_newrender(dataset, params, num_frames, eval_dir, sil_thres=config['mapping']['sil_thres'],
                                mapping_iters=config['mapping']['num_iters'], add_new_gaussians=config['mapping']['add_new_gaussians'],
                                eval_every=config['eval_every'], save_frames=False)
                    elif "tree" in dataset.sem_mode:
                        if flag_mlp_use ==1:
                            # show_semantic(dataset, params, num_frames, eval_dir, flag_mlp=flag_mlp_use, mlp_func=MLP_func)
                            eval_semantic_tree_newrender(dataset, params, num_frames, eval_dir, sil_thres=config['mapping']['sil_thres'],
                                mapping_iters=config['mapping']['num_iters'], add_new_gaussians=config['mapping']['add_new_gaussians'],
                                eval_every=config['eval_every'], save_frames=False, flag_mlp=flag_mlp_use, mlp_func=MLP_func)
                        else:
                            eval_semantic_tree_newrender(dataset, params, num_frames, eval_dir, sil_thres=config['mapping']['sil_thres'],
                                mapping_iters=config['mapping']['num_iters'], add_new_gaussians=config['mapping']['add_new_gaussians'],
                                eval_every=config['eval_every'], save_frames=True)
            else:
                eval_nvs(dataset, params, num_frames, eval_dir, sil_thres=config['mapping']['sil_thres'],
                    mapping_iters=config['mapping']['num_iters'], add_new_gaussians=config['mapping']['add_new_gaussians'],
                    eval_every=config['eval_every'], save_frames=True)
                
    time6 = time.time()
    print("5. eval main:  (s) (min) (h)", (time6-time5), (time6-time5)/60.0, (time6-time5)/3600.0)
    
    # Close WandB
    if config['use_wandb']:
        wandb_run.finish()

    time7 = time.time()
    print("6. close wandb: ", time7-time6, "s")

    # time summary
    print("======== time summary: ======== ")
    print("1. load experiments: ", time2-time1, "s")
    print("2. load dataset: (s) (min)", (time3-time2), (time3-time2)/60.0)
    print("3. load trained params: ", time4-time3, "s")
    print("4. wandb: ", time5-time4, "s")
    print("5. eval main: (s) (min) (h)", (time6-time5), (time6-time5)/60.0, (time6-time5)/3600.0)
    print("6. close wandb: ", time7-time6, "s")
