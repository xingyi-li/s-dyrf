import torch.optim
import json
import imageio
import os
import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm
from icecream import ic


from hexplane.model import init_model
from config.config import Config
from hexplane.dataloader import get_test_dataset, get_train_dataset
from hexplane.render.render import OctreeRender_trilinear_fast


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

ic("Preprocessing step")
cfg.style.dataset_type = style_dict["dataset_type"]
cfg.style.scene_name = style_dict["scene_name"]
cfg.style.style_name = style_dict["style_name"]
cfg.style.out_dir = os.path.join(cfg.style.out_dir, cfg.style.dataset_type, cfg.style.scene_name, cfg.style.style_name)

save_dir = os.path.join(cfg.style.out_dir, "ckpt_rgb_no_view_pt")
os.makedirs(save_dir, exist_ok=True)

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


W, H = train_dataset.img_wh

count = 0
frames = []
with torch.no_grad():
    for n in tqdm(range(len(train_dataset))):
        data = train_dataset[n]
        samples, gt_rgb, sample_times = data["rays"], data["rgbs"], data["time"]
        depth = None

        rays = samples.view(-1, samples.shape[-1])
        times = sample_times.view(-1, sample_times.shape[-1])

        rgb_map, _, _, _, _ = OctreeRender_trilinear_fast(
            rays,
            times,
            HexPlane,
            chunk=4096,
            N_samples=-1,
            ndc_ray=ndc_ray,
            white_bg=white_bg,
            device=device,
        )

        rgb_map = rgb_map.clamp(0.0, 1.0)
        rgb_map = rgb_map.view(H, W, 3).cpu()
        rgb_map = (rgb_map.numpy() * 255).astype("uint8")
        frames.append(rgb_map)
        count += 1
        if cfg.style.dataset_type == "neural3D_NDC":
            if count == 300:
                prefix = n // 300 + 1
                imageio.mimwrite(f"{save_dir}/cam{prefix:02d}.mp4", np.stack(frames), fps=30, quality=8)
                count = 0
                frames = []
        elif cfg.style.dataset_type == "dnerf":
            os.makedirs(f"{save_dir}/train", exist_ok=True)
            imageio.imwrite(f"{save_dir}/train/r_{count-1:03d}.png", rgb_map)

if cfg.style.dataset_type == "neural3D_NDC":
    frames = []
    with torch.no_grad():
        for n in tqdm(range(len(test_dataset))):
            data = test_dataset[n]
            samples, gt_rgb, sample_times = data["rays"], data["rgbs"], data["time"]
            depth = None

            rays = samples.view(-1, samples.shape[-1])
            times = sample_times.view(-1, sample_times.shape[-1])

            rgb_map, _, _, _, _ = OctreeRender_trilinear_fast(
                rays,
                times,
                HexPlane,
                chunk=4096,
                N_samples=-1,
                ndc_ray=ndc_ray,
                white_bg=white_bg,
                device=device,
            )

            rgb_map = rgb_map.clamp(0.0, 1.0)
            rgb_map = rgb_map.view(H, W, 3).cpu()
            rgb_map = (rgb_map.numpy() * 255).astype("uint8")
            frames.append(rgb_map)

        imageio.mimwrite(f"{save_dir}/cam00.mp4", np.stack(frames), fps=30, quality=8)
