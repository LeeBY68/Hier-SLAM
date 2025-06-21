# Hier-SLAM: Scaling-up Semantics in SLAM with a Hierarchically Categorical Gaussian Splatting (ICRA'25)

**Authors**: [Boying Li](https://leeby68.github.io/), [Zhixi Cai](https://profile.controlnet.space/), [Yuan-Fang Li](https://research.monash.edu/en/persons/yuan-fang-li), [Ian Reid](https://mbzuai.ac.ae/study/faculty/ian-reid/), and [Hamid Rezatofighi](https://vl4ai.erc.monash.edu/)

📝  [[Paper]](https://arxiv.org/abs/2409.12518)
&emsp;
📽️ [[Video]](https://youtu.be/Lp0QPMDTTHk)

<p align="center">
  <a href="">
    <img src="./assets/5.gif" alt="Logo" width="100%">
  </a>
</p>

We propose 🌳 Hier-SLAM: a **⭐ LLM-assitant ⭐ Fast ⭐ Semantic 3D Gaussian Splatting SLAM method** featuring **⭐ a Novel Hierarchical Categorical Representation**, which enables accurate global 3D semantic mapping, scaling-up capability, and explicit semantic prediction in the 3D world. 

## Hier-SLAM

- [Installation](#Installation)
- [Downloads](#Download)
- [Run](#Run)
  - [Replica](#replica)
  - [ScanNet](#scannet)
  - [Tree generation](#tree-generation)
- [Evaluation and visualization](#Evaluation-and-visualization)
- [Acknowledgement](#Acknowledgement)
- [Citation](#Citation)

# Getting Start

## Installation 
Clone the repository and set up the Conda environment:
```
git clone https://github.com/LeeBY68/Hier-SLAM.git
cd HierSLAM
conda create -n hierslam python=3.10
conda activate hierslam
conda install gcc=10 gxx=10 -c conda-forge
conda install -c "nvidia/label/cuda-11.6.0" cuda-toolkit
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge
pip install -r requirements.txt

# Compile the Semantic-3DGS：
cd hierslam-diff-gaussian-rasterization-w-depth
pip install ./
```

## Download

### Replica
The [Replica dataset](https://github.com/facebookresearch/Replica-Dataset) is a synthetic indoor dataset.
Our method uses the **same sequences** provided by previous works, including NICE-SLAM and iMAP (Same RGB and depth sequences with the same trajectories), to ensure a fair comparison with visual SLAM methods.

Since these sequences do not generate per-frame semantic ground truth, we have **rendered and generated the semantic ground truth** from the synthetic Replica dataset.
- To automatically download the Replica RGBD sequences, run the following script to download the data originally generated via NICE-SLAM:
  ```
  bash bash_scripts/download_replica.sh
  ```
- Download the corresponding per-frame semantic ground truth we rendered from the **following link**:
📥  [Replica_Semantic_Tree](https://monashuni-my.sharepoint.com/:f:/g/personal/boying_li_monash_edu/ElSCIy6TCVRIjeL5dMvX7a0BmIXTliIV56JIJr8Ku0mctw?e=RBxLCg) 

- The generated hierarchical tree file `info_semantic_tree.json`, located under the Replica directory. The tree is created based on the entire set of semantic classes in the Replica dataset (`info_semantic.json`: provided by official Replica). Copy `info_semantic_tree.json` into each sequence folder.

- After downloading the RGB & depth & poses & semantic gt, and the tree file, the final directory structure for Replica should be as follows (click to expand):

    <details>
      <summary>[Final Replica Structure]</summary>
    
    ```
      DATAROOT
      └── Replica
            └── room0
                ├── results
                │   ├── depth000000.png
                │   ├── depth000001.png
                │   ├── ...
                │   └── ...    
                │   ├── frame000000.png
                │   ├── frame000001.png
                │   ├── ...
                │   └── ...            
                ├── semantic_class
                │   ├── semantic_class_0.png
                │   ├── semantic_class_1.png
                │   ├── ...
                │   └── ... 
                └── traj.txt 
                └── info_semantic_tree.json 
    ```
    </details>

### ScanNet
The [ScanNet dataset](https://github.com/ScanNet/ScanNet) is a real-world RGB-D video dataset.
- To use it, follow the official data download procedure provided on the ScanNet website. After downloading, extract the color and depth frames from the `.sens` files using the [provided reader script](https://github.com/ScanNet/ScanNet/blob/master/SensReader/python/reader.py).
- For semantic labels, we use the `label-filt` files due to their higher quality:
  ```
  unzip [seq_name]_2d-label-filt.zip
  ```
- We provide the generated hierarchical tree files for ScanNet, which are all generated from the semantic classes in the ScanNet dataset (`scannetv2-labels.combined.tsv`: Provided by official ScanNet):

    - `scannetv2-labels.combined.tree.tsv`: A tree generated based on the NYU40 semantic classes.
    
    - `scannetv2-labels.combined.tree-large.tsv`: A large tree generated based on the full set of original ScanNet semantic classes, covering up to 550 unique labels, derived from the 'id' and 'category' columns in scannetv2-labels.combined.tsv.

    You can download both hierarchical tree files from the **following link**:
📥 [ScanNet_Tree](https://monashuni-my.sharepoint.com/:f:/g/personal/boying_li_monash_edu/EjTOn1JvMPVOuqiO5uwOSGwBQJ5abbjxgmYP8zqoCINuDA?e=aVj5y5)

- After downloading the RGB & depth & poses & semantics and the tree file, the final directory structure for ScanNet should be as follows (click to expand):
    <details>
      <summary>[Final ScanNet Structure]</summary>
    
    ```
      DATAROOT
      └── scannet
            └── scene0000_00
                ├── color
                │   ├── 0.jpg
                │   ├── 1.jpg
                │   ├── ...
                │   └── ...    
                ├── depth
                │   ├── 0.png
                │   ├── 1.png
                │   ├── ...
                │   └── ...            
                ├── label-filt
                │   ├── 0.png
                │   ├── 1.png
                │   ├── ...
                │   └── ... 
                ├── intrinsic
                └── pose
                    ├── 0.txt
                    ├── 1.txt
                    ├── ...
                    └── ... 
            ├── scanetv2-labels.combined.tree.tsv  
            └── scanetv2-labels.combined.tree-large.tsv    
    ```
    </details>


## Run

### Replica

🔹 Run Hier-SLAM on the Replica dataset using the default **hierarchical semantic setting**, use:
```bash
python scripts/hierslam.py configs/replica/hierslam_semantic_run.py
```

You can also try different configurations:

🔹 Run Hier-SLAM without semantic (**Visual-only Hier-SLAM**) on Replica, use:
  ```bash
  python scripts/hierslam.py configs/replica/hierslam_nosemantic_run.py
  ```
🔹 Run Hier-SLAM with **flat semantic encoding (one-hot)** :
  - Modify the number of semantic categories in the CUDA config file:
  ```cpp
  // In hierslam-diff-gaussian-rasterization-w-depth/cuda_rasterizer/config.h
  #define NUM_SEMANTIC 102
  ```
  - Reinstall the CUDA extension:
  ```bash
  conda activate hierslam
  cd hierslam-diff-gaussian-rasterization-w-depth
  pip install ./
  cd ..
  ```
  - Run following command:
```bash
 python scripts/hierslam.py configs/replica/hierslam_semantic_flat_run.py
  ```

### ScanNet
🔹 Run Hier-SLAM on the ScanNet dataset using the default **hierarchical semantic setting**:
- Modify the number of semantic categories in the CUDA config file:
```cpp
// In hierslam-diff-gaussian-rasterization-w-depth/cuda_rasterizer/config.h
#define NUM_SEMANTIC 16
```
- Reinstall the CUDA extension:
```bash
conda activate hierslam
cd hierslam-diff-gaussian-rasterization-w-depth
pip install ./
cd ..
```
- Run following command:
```bash
python scripts/hierslam.py configs/scannet/hierslam_semantic_run.py
```

You can also try different configurations:

🔹 Run Hier-SLAM without Semantic (**Visual-only Hier-SLAM**) on ScanNet, use:
```bash
python scripts/hierslam.py configs/scannet/hierslam_nosemantic_run.py
```

🔹 Run Hier-SLAM with **scaling-up semantic encoding** :
  - Modify the number of semantic categories in the CUDA config file:
  ```cpp
  // In hierslam-diff-gaussian-rasterization-w-depth/cuda_rasterizer/config.h
  #define NUM_SEMANTIC 74
  ```
  - Reinstall the CUDA extension:
  ```bash
  conda activate hierslam
  cd hierslam-diff-gaussian-rasterization-w-depth
  pip install ./
  cd ..
  ```
  - Run following command:
```bash
 python scripts/hierslam.py configs/scannet/hierslam_semantic_large_run.py
  ```


### Tree generation

Refer to [`LLM_tree/readme.md`](LLM_tree/readme.md) for details on tree generation using LLMs.

## Evaluation and visualization

🔸 Once a sequence completes, run following command to evaluate:
```bash
python scripts/eval_novel_view.py configs/replica/hierslam_semantic_run.py
```
- **Subset-classes evaluation**: In `configs/scannet/hierslam_semantic_run.py`, set `eval_gt_transfer = True` to evaluate only the classes visible in each frame. 
- We **recommend** using the full set of semantic classes (`eval_gt_transfer = False`) for a standard semantic evaluation. The subset option is provided to maintain consistency with previous works and ensure fair comparisons.

🔸 To export the reconstructed global 3D semantic world to a `.PLY` file by running:
```bash
python scripts/export_ply_semantic_tree.py configs/replica/hierslam_semantic_run.py
```
We recommend using MeshLab or Blender to visualize the resulting PLY files.

🔸 To visualize the reconstructed semantic map and estimated camera poses, run:
```bash
python viz_scripts/online_recon_sem_replica.py configs/replica/hierslam_semantic_run.py --flag_semantic
```
- Add `--flag_semantic` to enable semantic visualization. 
- Omit `--flag_semantic` to display the RGB reconstruction instead.

## Acknowledgement
We thank the authors for releasing code for their awesome works:
[3DGS](https://github.com/graphdeco-inria/gaussian-splatting)、
[SplaTAM](https://github.com/spla-tam/SplaTAM)、
[GauStudio](https://github.com/GAP-LAB-CUHK-SZ/gaustudio)、
[Gaussian Grouping](https://github.com/lkeab/gaussian-grouping)、
[Feature 3DGS](https://github.com/ShijieZhou-UCLA/feature-3dgs), and many other inspiring works in the community.

## Citation
If you find our work useful, please cite:
```
@inproceedings{li2025hier,
  title={Hier-SLAM: Scaling-up Semantics in SLAM with a Hierarchically Categorical Gaussian Splatting},
  author={Li, Boying and Cai, Zhixi and Li, Yuan-Fang and Reid, Ian and Rezatofighi, Hamid},
  booktitle={IEEE International Conference on Robotics and Automation (ICRA)},
  year={2025}
}
```





