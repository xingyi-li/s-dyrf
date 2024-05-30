import torch
import os
import shutil
from omegaconf import OmegaConf
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from icecream import ic

from hexplane.render.trainer import Trainer
from hexplane.model import init_model
from config.config import Config
from hexplane.dataloader import get_test_dataset, get_train_dataset
from hexplane.render.render import evaluation, evaluation_path


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

out_dir = os.path.dirname(cfg.systems.ckpt)

scene_name = cfg.data.datadir.split("/")[-1]
if cfg.systems.add_timestamp:
    logfolder = f'{cfg.systems.basedir}/{cfg.expname}/{scene_name}{datetime.now().strftime("-%Y%m%d-%H%M%S")}/finetune_no_view'
else:
    logfolder = f"{cfg.systems.basedir}/{cfg.expname}/{scene_name}/finetune_no_view"
os.makedirs(logfolder, exist_ok=True)

summary_writer = SummaryWriter(logfolder)


with open(os.path.join(logfolder, "cfg.yaml"), "w") as f:
    OmegaConf.save(config=cfg, f=f)
    # Changed name to prevent errors
    shutil.copyfile(__file__, os.path.join(logfolder, "opt_frozen.py"))

if cfg.data.datasampler_type == "rays":
    train_dataset = get_train_dataset(cfg, is_stack=False)
else:
    train_dataset = get_train_dataset(cfg, is_stack=True)
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

# init trainer.
trainer = Trainer(
    HexPlane,
    cfg,
    reso_cur,
    train_dataset,
    test_dataset,
    summary_writer,
    logfolder,
    device
)

trainer.train(cfg.optim.n_iters, cfg.optim.n_iters + 5000)
torch.save(HexPlane, os.path.join(logfolder, 'ckpt_rgb_no_view.pt'))