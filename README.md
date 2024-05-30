# S-DyRF: Reference-Based Stylized Radiance Fields for Dynamic Scenes (CVPR 2024)

[Xingyi Li](https://xingyi-li.github.io/)<sup>1,2</sup>,
[Zhiguo Cao](http://english.aia.hust.edu.cn/info/1085/1528.htm)<sup>1</sup>,
[Yizheng Wu](https://scholar.google.com/citations?user=0_iF4jMAAAAJ&hl=en)<sup>1,2</sup>,
[Kewei Wang](https://scholar.google.com/citations?user=fW7pUGMAAAAJ&hl=en)<sup>1,2</sup>,
[Ke Xian](https://kexianhust.github.io/)<sup>1,2\*</sup>,
[Zhe Wang](https://wang-zhe.me/)<sup>3</sup>,
[Guosheng Lin](https://guosheng.github.io/)<sup>2\*</sup>

<sup>1</sup>Huazhong University of Science and Technology, <sup>2</sup>S-Lab, Nanyang Technological University, <sup>3</sup>SenseTime Research

[Project](https://xingyi-li.github.io/s-dyrf/) | [Paper](https://github.com/xingyi-li/s-dyrf/) | [arXiv](https://arxiv.org/abs/2403.06205) | [Video](https://www.youtube.com/watch?v=1MFma4jRy9c&t=11s) | [Supp](https://github.com/xingyi-li/s-dyrf/) | [Poster](https://github.com/xingyi-li/s-dyrf/)

This repository contains the official PyTorch implementation of our CVPR 2024 paper "S-DyRF: Reference-Based Stylized Radiance Fields for Dynamic Scenes".

## Environment Setup

### 1. S-DyRF setup

```
git clone https://github.com/xingyi-li/s-dyrf.git
conda create -n s-dyrf python=3.8
conda activate s-dyrf
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
python setup.py develop
```

### 2. Video stylization method setup

As per the authors of "Interactive Video Stylization Using Few-Shot Patch-Based Training", all example commands and build scripts in this section assume Windows. Therefore, we setup this video stylization method on Windows to prepare training data, and Linux for model training.

```
git clone https://github.com/OndrejTexler/Few-Shot-Patch-Based-Training.git
conda create -n fspbt python=3.8
conda activate fspbt
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install opencv-python Pillow PyYAML scikit-image scipy tensorflow
```

## Data Preparation

Both [D-NeRF dataset](https://github.com/albertpumarola/D-NeRF) and [Plenoptic dataset](https://github.com/facebookresearch/Neural_3D_Video) can be downloaded from their official websites. Unzip and put them in the directory `dataset`. Please change the "datadir" in config based on the locations of downloaded datasets.

## Quick Start

### 0. Modify config file

Remember to modify systems.basedir and data.datadir in `config/nv3d/nv3d_${SCENE}.yaml`, `config/nv3d/nv3d_${SCENE}_style.yaml`, `config/dnerf/dnerf_${SCENE}.yaml`, and `config/dnerf/dnerf_${SCENE}_style.yaml`.

### 1. Pre-train HexPlane

To pre-train HexPlane on a specific scene, simply run

```bash
conda activate s-dyrf
python main.py config=config/nv3d/nv3d_${SCENE}.yaml # Plenoptic dataset
python main.py config=config/dnerf/dnerf_${SCENE}.yaml # D-NeRF dataset
```

where `${SCENE}` should be replaced by the name of the scene, e.g., `coffee_martini`, `cut_roasted_beef` for Plenoptic dataset, or `bouncingballs` for D-NeRF dataset.

You can also apply bash scripts to pre-train HexPlane on all scenes (modify the script if necessary):

```bash
bash scripts/train_hexplane_nv3d_all.sh # Plenoptic dataset
bash scripts/train_hexplane_dnerf_all.sh # D-NeRF dataset
```

### 2. Finetune HexPlane

To avoid geometry optimization and view-dependent stylization, we freeze density function and fix the view directions to zero. For Plenoptic dataset, we finetune Hexplane for additional steps:

```bash
conda activate s-dyrf
python main_finetune_no_view.py systems.ckpt="log/nv3d/${SCENE}/nv3d.th" config="config/nv3d/nv3d_${SCENE}.yaml"
```

Or simply run (modify the script if necessary)

```bash
bash scripts/finetune_hexplane_no_view_nv3d_all.sh # Plenoptic dataset
```

### 3. Generate stylized reference view

First you should render a reference view at specific time from a specific reference camera and then apply a 2D style transfer using an appropriate method, e.g., manual editing, [NNST](https://github.com/nkolkin13/NeuralNeighborStyleTransfer/tree/main), or [ControlNet](https://github.com/lllyasviel/ControlNet), to produce a stylized reference image. Then create data_config.json to provide information required. Some examples are provided in `dataset/ref_case/`.

### 4. Generate temporal pseudo-references

The process begins by rendering novel times that share an identical camera pose with the stylized reference image using the pre-trained dynamic radiance field. 

For D-NeRF dataset, it can be done by running:

``` bash
conda activate s-dyrf
python render_dnerf_specific_pose.py systems.ckpt="log/dnerf/${SCENE}/dnerf.th" style.ref_data_dir="dataset/ref_case/${STYLE}" config="config/dnerf/dnerf_${SCENE}_style.yaml"
```

Or simply run (modify the script if necessary)

```bash
bash scripts/render_dnerf_specific_pose_all.sh # D-NeRF dataset
```

For Plenoptic dataset, here we select the first image of the `imgs_test_all` folder within the Plenoptic dataset as the reference view. 

Next, we need to generate data for video stylization training:

```bash
# on Windows:
# i. cd to Few-Shot-Patch-Based-Training
# ii. put provided `tool_all.py` to `Few-Shot-Patch-Based-Training/_tools/`
# iii. modify _tools/tool_all.py, especially dataset_names, paths, data_path, FIRST, and LAST
# the default is to choose the first image as the reference view and put it in ${SCENE}_train, please modify the `copy files to train and gen` part of `tool_all.py` if necessary
# iv. run it, which will generate folders ${SCENE}_train and ${SCENE}_gen：
python _tools/tool_all.py
# v. copy ${SCENE}_train and ${SCENE}_gen to Linux and put them in directories `Few-Shot-Patch-Based-Training/data/${SCENE}_train` and `Few-Shot-Patch-Based-Training/data/${SCENE}_gen`
```

After that, train the video stylization method:

```bash
# on Linux:
# i. cd to Few-Shot-Patch-Based-Training
# ii. run Few-Shot-Patch-Based-Training/train.py to train the model:
conda activate fspbt
python train.py --config "_config/reference_P.yaml" --data_root "data/${SCENE}_train" --log_interval 1000 --log_folder logs_reference_P
# iii. generate temporal pseudo-references
python generate.py --checkpoint "data/${SCENE}_train/logs_reference_P/model_00020.pth" --data_root "data/${SCENE}_gen" --dir_input "input_filtered" --outdir "data/${SCENE}_gen/res_00020" --device "cuda:0"
# iv. move the result to /path/to/s-dyrf/dataset/ref_case/${STYLE}/style_time
mkdir /path/to/s-dyrf/dataset/ref_case/${STYLE}/style_time
cp data/${SCENE}_gen/res_00020/* /path/to/s-dyrf/dataset/ref_case/${STYLE}/style_time
```

where `${STYLE}` should be replaced by the style name, e.g., `coffee_martini_colorful`, `cut_roasted_beef_wave`, or `bouncingballs_colorful`.

Alternatively, once you have already prepared all data, you can simply run (modify the script if necessary)

```bash
bash train_fspbt_nv3d_all.sh # Plenoptic dataset
bash train_fspbt_dnerf_all.sh # D-NeRF dataset
```

### 5. Pre-processing step

```bash
# cd to s-dyrf
# activate conda environment
conda activate s-dyrf

# Plenoptic dataset
python ref_pre.py systems.ckpt="log/nv3d/${SCENE}/finetune_no_view/ckpt_rgb_no_view.pt" style.ref_data_dir="dataset/ref_case/${STYLE}" config="config/nv3d/nv3d_${SCENE}_style.yaml"

python ref_regist_time.py systems.ckpt="log/nv3d/${SCENE}/finetune_no_view/ckpt_rgb_no_view.pt" style.ref_data_dir="dataset/ref_case/${STYLE}" config="config/nv3d/nv3d_${SCENE}_style.yaml"

# D-NeRF dataset
python ref_pre.py systems.ckpt="log/dnerf/${SCENE}/dnerf.th" style.ref_data_dir="dataset/ref_case/${STYLE}" config="config/dnerf/dnerf_${SCENE}_style.yaml"
python ref_regist_time.py systems.ckpt="log/dnerf/${SCENE}/dnerf.th" style.ref_data_dir="dataset/ref_case/${STYLE}" config="config/dnerf/dnerf_${SCENE}_style.yaml"
```

Or simply run (modify the script if necessary)

```bash
bash preprocess_nv3d_all.sh # Plenoptic dataset
bash preprocess_dnerf_all.sh # D-NeRF dataset
```

### 6. Train S-DyRF & render stylized video

```bash
# activate conda environment
conda activate s-dyrf

# Plenoptic dataset
cp dataset/neural_3D/${SCENE}/poses_bounds.npy stylize/neural3D_NDC/${SCENE}/${STYLE}/ckpt_rgb_no_view_pt/
python ref_style_time.py expname=${EXPNAME} systems.ckpt="log/nv3d/${SCENE}/finetune_no_view/ckpt_rgb_no_view.pt" style.ref_data_dir="dataset/ref_case/${STYLE}" config="config/nv3d/nv3d_${SCENE}_style.yaml"
python main.py config="config/nv3d/nv3d_${SCENE}_style.yaml" style.ref_data_dir="dataset/ref_case/${STYLE}" systems.ckpt="stylize/neural3D_NDC/${SCENE}/${STYLE}/${EXPNAME}/ckpt_style.pt" render_only=True render_specific_time=True

# D-NeRF dataset
cp dataset/dnerf/${SCENE}/transforms_train.json stylize/dnerf/${SCENE}/${STYLE}/ckpt_rgb_no_view_pt/
python ref_style_time.py expname=${EXPNAME} systems.ckpt="log/dnerf/${SCENE}/dnerf.th" style.ref_data_dir="dataset/ref_case/${STYLE}" config="config/dnerf/dnerf_${SCENE}_style.yaml"
python main.py config="config/dnerf/dnerf_${SCENE}_style.yaml" style.ref_data_dir="dataset/ref_case/${STYLE}" systems.ckpt="stylize/dnerf/${SCENE}/${STYLE}/${EXPNAME}/ckpt_style.pt" render_only=True render_specific_pose=True render_specific_time=True
```

Or simply run (modify the script if necessary)

```bash
bash stylize_hexplane_nv3d_all.sh # Plenoptic dataset
bash stylize_hexplane_dnerf_all.sh # D-NeRF dataset
```

## Citation

If you find our work useful in your research, please consider to cite our paper:
```
@article{li2024sdyrf,
    title={S-DyRF: Reference-Based Stylized Radiance Fields for Dynamic Scenes},
    author={Li, Xingyi and Cao, Zhiguo and Wu, Yizheng and Wang, Kewei and Xian, Ke and Wang, Zhe and Lin, Guosheng},
    journal={arXiv preprint arXiv:2403.06205},
    year={2024}
}
```

## Acknowledgement

This code is built on [HexPlane](https://github.com/Caoang327/HexPlane), [Ref-NPR](https://github.com/dvlab-research/Ref-NPR) and many other projects. We would like to acknowledge them for making great code openly available for us to use.
