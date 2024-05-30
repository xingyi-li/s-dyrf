import torch
from omegaconf import OmegaConf
import json
import os

from config.config import Config
from hexplane.dataloader import get_test_dataset, get_train_dataset
from hexplane.render.render import evaluation_specific_pose


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float32)


def render_specific_pose(cfg):
    test_dataset = get_test_dataset(cfg, is_stack=True)
    ndc_ray = test_dataset.ndc_ray
    white_bg = test_dataset.white_bg

    if not os.path.exists(cfg.systems.ckpt):
        print("the ckpt path does not exists!!")
        return

    HexPlane = torch.load(cfg.systems.ckpt, map_location=device)
    if cfg.stylize:
        HexPlane.turn_off_viewdep()

        # freeze density function
        HexPlane.freeze_density()
    logfolder = os.path.dirname(cfg.systems.ckpt)


    os.makedirs(f"{logfolder}/imgs_specific_pose", exist_ok=True)
    evaluation_specific_pose(
        test_dataset,
        HexPlane,
        cfg,
        f"{logfolder}/imgs_specific_pose/",
        prefix="test",
        N_vis=-1,
        N_samples=-1,
        ndc_ray=ndc_ray,
        white_bg=white_bg,
        device=device,
    )



if __name__ == "__main__":
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

    # # Fix Random Seed for Reproducibility.
    # random.seed(cfg.systems.seed)
    # np.random.seed(cfg.systems.seed)
    # torch.manual_seed(cfg.systems.seed)
    # torch.cuda.manual_seed(cfg.systems.seed)

    data_dir = cfg.style.ref_data_dir
    if data_dir == "":
        print("Please specify the ref data directory")
        exit()

    # load json file.
    with open(os.path.join(data_dir, "data_config.json")) as fp:
        style_dict = json.load(fp)

    cfg.style.tmpl_idx_test = style_dict["tmpl_idx_test"]
    cfg.style.scene_name = style_dict["scene_name"]

    render_specific_pose(cfg)
