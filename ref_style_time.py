import torch.optim
import json
import os
from os import path
import shutil
import numpy as np
from omegaconf import OmegaConf
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import cv2
from icecream import ic

from hexplane.render.style_trainer_time import StyleTrainer
from hexplane.model import init_model
from config.config import Config
from hexplane.dataloader import get_test_dataset, get_train_dataset


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float32)

# Load config file from base config, yaml and cli.
base_cfg = OmegaConf.structured(Config())
cli_cfg = OmegaConf.from_cli()
base_yaml_path = base_cfg.get("config", None)
yaml_path = cli_cfg.get("config", None)
if yaml_path is not None:
    yaml_cfg = OmegaConf.load(yaml_path)
elif base_yaml_path is not None:
    yaml_cfg = OmegaConf.load(base_yaml_path)
else:
    yaml_cfg = OmegaConf.create()
cfg = OmegaConf.merge(base_cfg, yaml_cfg, cli_cfg)  # merge configs

data_dir = cfg.style.ref_data_dir
with open(os.path.join(data_dir, "data_config.json")) as fp:
    style_dict = json.load(fp)

ic("Optimization step")
cfg.style.color_pre = style_dict["color_pre"]
cfg.style.dataset_type = style_dict["dataset_type"]
cfg.style.scene_name = style_dict["scene_name"]
cfg.style.style_img = style_dict["style_img"]

if cfg.style.style_multiview:
    cfg.style.style_img_multiview = style_dict["style_img_multiview"]
    cfg.style.multiview_img_cam_idx = style_dict["multiview_img_cam_idx"]

# include temporal pseudo-references
from glob import glob
stylized_dir = os.path.join(os.path.dirname(cfg.style.style_img[0]), "style_time")
tmp = glob(os.path.join(stylized_dir, "*.png"))
tmp = sorted(tmp, key=lambda i: int(os.path.splitext(os.path.basename(i))[0]))
if cfg.style.dataset_type == "neural3D_NDC":
    for i in range(len(cfg.style.style_img)):
        tmp[style_dict["tmpl_idx_test"][i]] = cfg.style.style_img[i]
elif cfg.style.dataset_type == "dnerf":
    tmp[style_dict["tmpl_idx"][0]] = cfg.style.style_img[0]
cfg.style.style_img = tmp

cfg.style.style_name = style_dict["style_name"]
cfg.style.tmpl_idx_test = style_dict["tmpl_idx_test"]
cfg.style.tmpl_idx_train = style_dict["tmpl_idx_train"]
cfg.style.ref_data_dir = "./dataset/neural_3D/{}".format(
    cfg.style.scene_name) if cfg.style.dataset_type == 'neural3D_NDC' else "./dataset/dnerf/{}".format(
    cfg.style.scene_name)

cfg.style.out_dir = os.path.join(cfg.style.out_dir, cfg.style.dataset_type, cfg.style.scene_name, cfg.style.style_name)
cfg.style.exchange_tmp = True
cfg.style.fast = False

ic(style_dict)
ic(cfg.style.vgg_blocks)

os.makedirs(cfg.style.out_dir, exist_ok=True)

logfolder = os.path.join(cfg.style.out_dir, cfg.expname)

os.makedirs(logfolder, exist_ok=True)

summary_writer = SummaryWriter(logfolder)


with open(path.join(logfolder, "cfg.yaml"), "w") as f:
    OmegaConf.save(config=cfg, f=f)
    # Changed name to prevent errors
    shutil.copyfile(__file__, path.join(logfolder, "opt_frozen.py"))


cur_datadir = cfg.data.datadir
cfg.data.datadir = os.path.join(cfg.style.out_dir, 'ckpt_rgb_no_view_pt')
if cfg.data.datasampler_type == "rays":
    cfg.data.datasampler_type = "images"
    train_dataset = get_train_dataset(cfg, is_stack=True)
else:
    train_dataset = get_train_dataset(cfg, is_stack=True)
cfg.data.datadir = cur_datadir
test_dataset = get_test_dataset(cfg, is_stack=True)
ndc_ray = test_dataset.ndc_ray
white_bg = test_dataset.white_bg
near_far = test_dataset.near_far

assert os.path.isfile(cfg.systems.ckpt), "must specify a initial checkpoint"
# init model.
aabb = train_dataset.scene_bbox.to(device)
HexPlane, reso_cur = init_model(cfg, aabb, near_far, device)
ic("Loaded ckpt: ", cfg.systems.ckpt)

# reinitialize appearance function
# discard view-dependent rendering and apply a view-independent fitting on training views
HexPlane.turn_off_viewdep()

# freeze density function
HexPlane.freeze_density()


###### resize style image such that its long side matches the long side of content images
style_imgs = [cv2.cvtColor(cv2.imread(cfg.style.style_img[i], 1), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0 for i in
              range(len(cfg.style.style_img))]

W, H = train_dataset.img_wh

for i in range(len(style_imgs)):
    style_imgs[i] = cv2.resize(style_imgs[i], (W, H), interpolation=cv2.INTER_CUBIC)

style_h, style_w = style_imgs[0].shape[:2]
content_long_side = max([W, H])


if cfg.style.fast:
    style_imgs_d = [cv2.resize(m, (style_w // 2, style_h // 2), cv2.INTER_AREA) for m in style_imgs]
    style_imgs_o = [torch.from_numpy(m).to(device=device) for m in style_imgs]
    style_imgs = [torch.from_numpy(m).to(device=device) for m in style_imgs_d]
else:
    style_imgs_o = [torch.from_numpy(m).to(device=device) for m in style_imgs]
    style_imgs = style_imgs_o

global_start_time = datetime.now()

epoch_id = 0
epoch_size = None
batches_per_epoch = None
batch_size = None


if cfg.style.color_pre:
    cfg.style.loss_names = ["tcm_loss", "color_patch", "content_loss"]
    cfg.style.vgg_blocks = [2, 3, 4]
else:
    cfg.style.loss_names = ["tcm_loss"]
    cfg.style.vgg_blocks = [2, 3, 4]


if cfg.style.dataset_type == "neural3D_NDC":
    tmpl_ids = list(range(300))
elif cfg.style.dataset_type == "dnerf":
    tmpl_ids = list(range(120))

# init trainer.
trainer = StyleTrainer(
    HexPlane,
    cfg,
    reso_cur,
    train_dataset,
    test_dataset,
    summary_writer,
    logfolder,
    device,
    style_imgs_o,
    tmpl_ids,
)

if cfg.style.dataset_type == "neural3D_NDC":
    trainer.train(655000, "style")
elif cfg.style.dataset_type == "dnerf":
    trainer.train(25000, "style")
torch.save(HexPlane, os.path.join(logfolder, 'ckpt_style.pt'))
