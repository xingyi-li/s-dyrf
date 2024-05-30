import datetime
import os
import random

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter

from config.config import Config
from hexplane.dataloader import get_test_dataset, get_train_dataset
from hexplane.model import init_model
from hexplane.render.render import evaluation, evaluation_path, evaluation_specific_pose, evaluation_specific_time
from hexplane.render.trainer import Trainer
from hexplane.render.util.Sampling import cal_n_samples

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float32)


def render_test(cfg):
    test_dataset = get_test_dataset(cfg, is_stack=True)
    ndc_ray = test_dataset.ndc_ray
    white_bg = test_dataset.white_bg

    if not os.path.exists(cfg.systems.ckpt):
        print("the ckpt path does not exists!!")
        return

    aabb = test_dataset.scene_bbox.to(device)
    near_far = test_dataset.near_far

    color_tf = None
    HexPlane, reso_cur = init_model(cfg, aabb, near_far, device)
    if hasattr(HexPlane, "color_tf"):
        color_tf = HexPlane.color_tf.cpu()

    if cfg.style.dataset_type == "neural3D_NDC":
        nSamples = min(
            cfg.model.nSamples,
            cal_n_samples(reso_cur, cfg.model.step_ratio),
        )
    elif cfg.style.dataset_type == "dnerf":
        nSamples = -1

    if cfg.stylize:
        HexPlane.turn_off_viewdep()

        # freeze density function
        HexPlane.freeze_density()
    logfolder = os.path.dirname(cfg.systems.ckpt)

    if cfg.render_train:
        import json
        data_dir = cfg.style.ref_data_dir
        if data_dir == "":
            print("Please specify the ref data directory")
            exit()
        # load json file.
        with open(os.path.join(data_dir, "data_config.json")) as fp:
            style_dict = json.load(fp)

        cfg.style.tmpl_idx_test = style_dict["tmpl_idx_test"]
        cfg.style.dataset_type = style_dict["dataset_type"]

        os.makedirs(f"{logfolder}/imgs_train_all", exist_ok=True)
        train_dataset = get_train_dataset(cfg, is_stack=True)
        evaluation(
            train_dataset,
            HexPlane,
            cfg,
            f"{logfolder}/imgs_train_all/",
            prefix="train",
            N_vis=-1,
            N_samples=nSamples,
            ndc_ray=ndc_ray,
            white_bg=white_bg,
            device=device,
            color_tf=color_tf,
        )

    if cfg.render_test:
        import json
        data_dir = cfg.style.ref_data_dir
        if data_dir == "":
            print("Please specify the ref data directory")
            exit()
        # load json file.
        with open(os.path.join(data_dir, "data_config.json")) as fp:
            style_dict = json.load(fp)

        cfg.style.tmpl_idx_test = style_dict["tmpl_idx_test"]
        cfg.style.dataset_type = style_dict["dataset_type"]

        os.makedirs(f"{logfolder}/imgs_test_all", exist_ok=True)
        evaluation(
            test_dataset,
            HexPlane,
            cfg,
            f"{logfolder}/imgs_test_all/",
            prefix="test",
            N_vis=-1,
            N_samples=nSamples,
            ndc_ray=ndc_ray,
            white_bg=white_bg,
            device=device,
            color_tf=color_tf,
        )

    if cfg.render_path:
        import json
        data_dir = cfg.style.ref_data_dir
        if data_dir == "":
            print("Please specify the ref data directory")
            exit()
        # load json file.
        with open(os.path.join(data_dir, "data_config.json")) as fp:
            style_dict = json.load(fp)

        cfg.style.tmpl_idx_test = style_dict["tmpl_idx_test"]
        cfg.style.dataset_type = style_dict["dataset_type"]

        os.makedirs(f"{logfolder}/imgs_path_all", exist_ok=True)
        evaluation_path(
            test_dataset,
            HexPlane,
            cfg,
            f"{logfolder}/imgs_path_all/",
            prefix="test",
            N_vis=-1,
            N_samples=nSamples,
            ndc_ray=ndc_ray,
            white_bg=white_bg,
            device=device,
            color_tf=color_tf,
        )

    if cfg.render_specific_pose:
        import json
        data_dir = cfg.style.ref_data_dir
        if data_dir == "":
            print("Please specify the ref data directory")
            exit()
        # load json file.
        with open(os.path.join(data_dir, "data_config.json")) as fp:
            style_dict = json.load(fp)

        cfg.style.tmpl_idx_test = style_dict["tmpl_idx_test"]
        cfg.style.dataset_type = style_dict["dataset_type"]

        os.makedirs(f"{logfolder}/imgs_specific_pose", exist_ok=True)
        evaluation_specific_pose(
            test_dataset,
            HexPlane,
            cfg,
            f"{logfolder}/imgs_specific_pose/",
            prefix="test",
            N_vis=-1,
            N_samples=nSamples,
            ndc_ray=ndc_ray,
            white_bg=white_bg,
            device=device,
            color_tf=color_tf,
        )

    if cfg.render_specific_time:
        import json
        data_dir = cfg.style.ref_data_dir
        if data_dir == "":
            print("Please specify the ref data directory")
            exit()
        # load json file.
        with open(os.path.join(data_dir, "data_config.json")) as fp:
            style_dict = json.load(fp)

        cfg.style.tmpl_idx_test = style_dict["tmpl_idx_test"]
        cfg.style.dataset_type = style_dict["dataset_type"]

        os.makedirs(f"{logfolder}/imgs_specific_time", exist_ok=True)
        evaluation_specific_time(
            test_dataset,
            HexPlane,
            cfg,
            f"{logfolder}/imgs_specific_time/",
            prefix="test",
            N_vis=-1,
            N_samples=nSamples,
            ndc_ray=ndc_ray,
            white_bg=white_bg,
            device=device,
            color_tf=color_tf,
        )


def reconstruction(cfg):
    if cfg.data.datasampler_type == "rays":
        train_dataset = get_train_dataset(cfg, is_stack=False)
    else:
        train_dataset = get_train_dataset(cfg, is_stack=True)
    test_dataset = get_test_dataset(cfg, is_stack=True)
    ndc_ray = test_dataset.ndc_ray
    white_bg = test_dataset.white_bg
    near_far = test_dataset.near_far

    scene_name = cfg.data.datadir.split("/")[-1]
    if cfg.systems.add_timestamp:
        logfolder = f'{cfg.systems.basedir}/{cfg.expname}/{scene_name}{datetime.datetime.now().strftime("-%Y%m%d-%H%M%S")}'
    else:
        logfolder = f"{cfg.systems.basedir}/{cfg.expname}/{scene_name}"

    # init log file
    os.makedirs(logfolder, exist_ok=True)
    os.makedirs(f"{logfolder}/imgs_vis", exist_ok=True)
    summary_writer = SummaryWriter(os.path.join(logfolder, "logs"))
    cfg_file = os.path.join(f"{logfolder}", "cfg.yaml")
    with open(cfg_file, "w") as f:
        OmegaConf.save(config=cfg, f=f)

    # init model.
    aabb = train_dataset.scene_bbox.to(device)
    HexPlane, reso_cur = init_model(cfg, aabb, near_far, device)
    nSamples = min(
        cfg.model.nSamples,
        cal_n_samples(reso_cur, cfg.model.step_ratio),
    )

    # init trainer.
    trainer = Trainer(
        HexPlane,
        cfg,
        reso_cur,
        train_dataset,
        test_dataset,
        summary_writer,
        logfolder,
        device,
    )

    trainer.train()

    torch.save(HexPlane, f"{logfolder}/{cfg.expname}.th")
    # Render training viewpoints.
    if cfg.render_train:
        os.makedirs(f"{logfolder}/imgs_train_all", exist_ok=True)
        train_dataset = get_train_dataset(cfg, is_stack=True)
        evaluation(
            train_dataset,
            HexPlane,
            cfg,
            f"{logfolder}/imgs_train_all/",
            prefix="train",
            N_vis=-1,
            N_samples=-1,
            ndc_ray=ndc_ray,
            white_bg=white_bg,
            device=device,
        )

    # Render test viewpoints.
    if cfg.render_test:
        os.makedirs(f"{logfolder}/imgs_test_all", exist_ok=True)
        evaluation(
            test_dataset,
            HexPlane,
            cfg,
            f"{logfolder}/imgs_test_all/",
            prefix="test",
            N_vis=-1,
            N_samples=-1,
            ndc_ray=ndc_ray,
            white_bg=white_bg,
            device=device,
        )

    # Render validation viewpoints.
    if cfg.render_path:
        os.makedirs(f"{logfolder}/imgs_path_all", exist_ok=True)
        evaluation_path(
            test_dataset,
            HexPlane,
            cfg,
            f"{logfolder}/imgs_path_all/",
            prefix="validation",
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

    # Fix Random Seed for Reproducibility.
    random.seed(cfg.systems.seed)
    np.random.seed(cfg.systems.seed)
    torch.manual_seed(cfg.systems.seed)
    torch.cuda.manual_seed(cfg.systems.seed)

    if cfg.render_only and (cfg.render_test or cfg.render_path):
        # Inference only.
        render_test(cfg)
    else:
        # Reconstruction and Inference.
        reconstruction(cfg)
