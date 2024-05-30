import cv2
import json
from tqdm import tqdm
from icecream import ic
import os
import numpy as np
import torch
from omegaconf import OmegaConf

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

cfg.style.color_pre = style_dict["color_pre"]
cfg.style.dataset_type = style_dict["dataset_type"]
cfg.style.scene_name = style_dict["scene_name"]
cfg.style.style_img = style_dict["style_img"]

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

os.makedirs(cfg.style.out_dir, exist_ok=True)

ic("Start ray registeration step")
if os.path.exists(os.path.join(cfg.style.out_dir, "color_corr")):
    ic("preload the correspondence, goto the next step")
    PRELOAD = True
    exit()
else:
    dict_dir = os.path.join(cfg.style.out_dir, "color_corr")
    os.makedirs(dict_dir, exist_ok=True)
ic(style_dict)

style_imgs = [cv2.cvtColor(cv2.imread(cfg.style.style_img[i], 1), cv2.COLOR_BGR2RGB) / 255.0 for i in
              range(len(cfg.style.style_img))]

HexPlane = torch.load(cfg.systems.ckpt, map_location=device)
# reinitialize appearance function
# discard view-dependent rendering and apply a view-independent fitting on training views
HexPlane.turn_off_viewdep()

# freeze density function
HexPlane.freeze_density()
ic("Loaded ckpt: ", cfg.systems.ckpt)
train_dataset = get_train_dataset(cfg, is_stack=True)
scene_bbox_min = train_dataset.scene_bbox[0]
scene_bbox_max = train_dataset.scene_bbox[1]
test_dataset = get_test_dataset(cfg, is_stack=True)
W, H = train_dataset.img_wh

for i in range(len(style_imgs)):
    style_imgs[i] = cv2.resize(style_imgs[i], (W, H), interpolation=cv2.INTER_CUBIC)

tmpl_ids = list(range(len(cfg.style.style_img)))

with torch.no_grad():
    for tmpl_id in tqdm(tmpl_ids):
        # Here, what we want to do is establish a fine lookup table between
        # the training data and the stylized templates. This can be divided into the following three steps:

        # Step A: First, pre-store all the information we need: depth, position, and position range.
        eps = 1e-5

        depths = []
        xyzs = []
        xyz_max = torch.zeros(3)
        xyz_min = torch.ones(3)

        if cfg.style.dataset_type == "neural3D_NDC":
            cur_time_idx_list = list(range(tmpl_id, len(train_dataset), 300))
            for n in tqdm(cur_time_idx_list):
                data = train_dataset[n]
                samples, gt_rgb, sample_times = data["rays"], data["rgbs"], data["time"]
                depth = None

                rays = samples.view(-1, samples.shape[-1])
                times = sample_times.view(-1, sample_times.shape[-1])

                rgb_map, _, depth_map, _, _ = OctreeRender_trilinear_fast(
                    rays,
                    times,
                    HexPlane,
                    chunk=4096,
                    N_samples=-1,
                    ndc_ray=train_dataset.ndc_ray,
                    white_bg=train_dataset.white_bg,
                    device=device,
                )
                depth_img = depth_map.reshape(H, W).cpu()[..., None]

                depth_rep = depth_img.repeat(1, 1, 3)
                xyz_pos = rays[..., :3].view(H, W, 3) + \
                          rays[..., 3:].view(H, W, 3) * depth_rep

                xyz_pos[..., 0] = torch.clamp(xyz_pos[..., 0], scene_bbox_min[0], scene_bbox_max[0])
                xyz_pos[..., 1] = torch.clamp(xyz_pos[..., 1], scene_bbox_min[1], scene_bbox_max[1])
                xyz_pos[..., 2] = torch.clamp(xyz_pos[..., 2], scene_bbox_min[2], scene_bbox_max[2])

                xyz_min = torch.minimum(xyz_min, torch.min(xyz_pos[depth_rep != 0].reshape(-1, 3), dim=0)[0])
                xyz_max = torch.maximum(xyz_max, torch.max(xyz_pos[depth_rep != 0].reshape(-1, 3), dim=0)[0])
                xyz_pos[depth_rep == 0] = 0
                depths.append(depth_img.unsqueeze(0))
                xyzs.append(xyz_pos.unsqueeze(0))

            depths = torch.cat(depths).cpu().numpy().reshape(len(cur_time_idx_list), H, W)
            xyzs = torch.cat(xyzs).cpu().numpy().reshape(len(cur_time_idx_list), H, W, -1)
            aabb = torch.cat((xyz_min, xyz_max)).cpu().numpy()
            ic(aabb)

            # Step B. For each point, we can find its corresponding geometric position,
            # and store template information
            # (xyz + direction, somewhat similar to the approach used in visualization tools) in a 3D grid (256^3).
            n_grid = 512
            grid_unit = 1.0 / n_grid
            grid_offset = 1.0 / n_grid * 0.5
            eps = 1e-5

            xyz_min = aabb[:3]
            xyz_max = aabb[3:]

            xyzs = (xyzs - xyz_min) / (xyz_max - xyz_min + eps)
            tmpl_xyzs = []
            tmpl_depths = []
            tmpl_dirs = []
            tmpl_ignore = []
            
            if cfg.style.tmpl_idx_train is not None:
                # not implemented yet.
                tmpl_xyzs += [xyzs[tmpl_id].reshape(-1, 3)]
                depth_img = depths[tmpl_id].reshape(-1)
                tmpl_depths += [depth_img]
                tmpl_dirs += [dirs[tmpl_id]]
            else:
                # with the test dataset.
                data = test_dataset[tmpl_id]
                samples, gt_rgb, sample_times = data["rays"], data["rgbs"], data["time"]
                depth = None

                rays = samples.view(-1, samples.shape[-1])
                times = sample_times.view(-1, sample_times.shape[-1])

                _, _, depth_map, _, _ = OctreeRender_trilinear_fast(
                    rays,
                    times,
                    HexPlane,
                    chunk=4096,
                    N_samples=-1,
                    ndc_ray=train_dataset.ndc_ray,
                    white_bg=train_dataset.white_bg,
                    device=device,
                )
                depth_img = depth_map.reshape(H, W).cpu()[..., None]

                depth_rep = depth_img.repeat(1, 1, 3)
                depth_img = depth_img.reshape(-1).cpu().numpy()

                tmpl_xyz = rays[..., :3].view(H, W, 3) + rays[..., 3:].view(H, W, 3) * depth_rep
                tmpl_xyz[..., 0] = torch.clamp(tmpl_xyz[..., 0], scene_bbox_min[0], scene_bbox_max[0])
                tmpl_xyz[..., 1] = torch.clamp(tmpl_xyz[..., 1], scene_bbox_min[1], scene_bbox_max[1])
                tmpl_xyz[..., 2] = torch.clamp(tmpl_xyz[..., 2], scene_bbox_min[2], scene_bbox_max[2])
                tmpl_xyz = tmpl_xyz.cpu().numpy()
                tmpl_xyz = (tmpl_xyz - xyz_min) / (xyz_max - xyz_min + eps)

                tmpl_dirs += (rays[..., 3:].reshape(-1, 3))
                tmpl_depths += [depth_img]
                tmpl_xyzs += [tmpl_xyz.reshape(-1, 3)]

            border_ignore_mask = depth_img.reshape(H, W)
            tmpl_ignore += [border_ignore_mask.reshape(-1)]

            tmpl_xyz = np.concatenate(tmpl_xyzs, axis=0)
            tmpl_depth = np.concatenate(tmpl_depths, axis=0)
            tmpl_dir = torch.concat(tmpl_dirs, dim=0)
            if len(tmpl_dir.shape) == 1:
                tmpl_dir = tmpl_dir.view(-1, 3)
            tmpl_ignore = np.concatenate(tmpl_ignore, axis=0)
            ic(tmpl_xyz.shape, tmpl_depth.shape, tmpl_dir.shape, tmpl_ignore.shape)

            tmpl_xyz = torch.from_numpy(tmpl_xyz).cuda()

        elif cfg.style.dataset_type == "dnerf":
            cur_time_idx_list = list(range(len(train_dataset)))
            sample_times = torch.FloatTensor([(tmpl_id) / 119. * 2.0 - 1.0]) * test_dataset.time_scale
            for n in tqdm(cur_time_idx_list):
                data = train_dataset[n]
                samples = data["rays"]
                times = sample_times.expand(samples.shape[0], 1)
                depth = None

                rays = samples.view(-1, samples.shape[-1])
                times = times.view(-1, times.shape[-1])

                rgb_map, _, depth_map, _, acc_map = OctreeRender_trilinear_fast(
                    rays,
                    times,
                    HexPlane,
                    chunk=4096,
                    N_samples=-1,
                    ndc_ray=train_dataset.ndc_ray,
                    white_bg=train_dataset.white_bg,
                    device=device,
                )
                depth_img = depth_map.reshape(H, W).cpu()[..., None]

                depth_img = depth_img * acc_map.reshape(H, W).cpu()[..., None]
                depth_rep = depth_img.repeat(1, 1, 3)
                xyz_pos = rays[..., :3].view(H, W, 3) + \
                          rays[..., 3:].view(H, W, 3) * depth_rep

                xyz_min = torch.minimum(xyz_min, torch.min(xyz_pos[depth_rep != 0].reshape(-1, 3), dim=0)[0])
                xyz_max = torch.maximum(xyz_max, torch.max(xyz_pos[depth_rep != 0].reshape(-1, 3), dim=0)[0])
                xyz_pos[depth_rep == 0] = 0
                depths.append(depth_img.unsqueeze(0))
                xyzs.append(xyz_pos.unsqueeze(0))

            depths = torch.cat(depths).cpu().numpy().reshape(len(cur_time_idx_list), H, W)
            xyzs = torch.cat(xyzs).cpu().numpy().reshape(len(cur_time_idx_list), H, W, -1)
            aabb = torch.cat((xyz_min, xyz_max)).cpu().numpy()
            ic(aabb)

            # Step B. For each point, we can find its corresponding geometric position,
            # and store template information
            # (xyz + direction, somewhat similar to the approach used in visualization tools) in a 3D grid (256^3).
            n_grid = 512
            grid_unit = 1.0 / n_grid
            grid_offset = 1.0 / n_grid * 0.5
            eps = 1e-5

            xyz_min = aabb[:3]
            xyz_max = aabb[3:]

            xyzs = (xyzs - xyz_min) / (xyz_max - xyz_min + eps)
            tmpl_xyzs = []
            tmpl_depths = []
            tmpl_dirs = []
            tmpl_ignore = []

            if cfg.style.tmpl_idx_train is not None:
                # not implemented yet.
                tmpl_xyzs += [xyzs[tmpl_id].reshape(-1, 3)]
                depth_img = depths[tmpl_id].reshape(-1)
                tmpl_depths += [depth_img]
                tmpl_dirs += [dirs[tmpl_id]]
            else:
                # with the test dataset.
                data = test_dataset[cfg.style.tmpl_idx_test[0]]
                samples = data["rays"]
                times = sample_times.expand(samples.shape[0], 1)
                depth = None

                rays = samples.view(-1, samples.shape[-1])
                times = times.view(-1, times.shape[-1])

                _, _, depth_map, _, acc_map = OctreeRender_trilinear_fast(
                    rays,
                    times,
                    HexPlane,
                    chunk=4096,
                    N_samples=-1,
                    ndc_ray=train_dataset.ndc_ray,
                    white_bg=train_dataset.white_bg,
                    device=device,
                )
                depth_img = depth_map.reshape(H, W).cpu()[..., None]

                depth_img = depth_img * acc_map.reshape(H, W).cpu()[..., None]
                depth_rep = depth_img.repeat(1, 1, 3)
                depth_img = depth_img.reshape(-1).cpu().numpy()

                tmpl_xyz = rays[..., :3].view(H, W, 3) + rays[..., 3:].view(H, W, 3) * depth_rep
                tmpl_xyz = tmpl_xyz.cpu().numpy()
                tmpl_xyz = (tmpl_xyz - xyz_min) / (xyz_max - xyz_min + eps)

                tmpl_dirs += (rays[..., 3:].reshape(-1, 3))
                tmpl_depths += [depth_img]
                tmpl_xyzs += [tmpl_xyz.reshape(-1, 3)]

            border_ignore_mask = depth_img.reshape(H, W)
            tmpl_ignore += [border_ignore_mask.reshape(-1)]

            tmpl_xyz = np.concatenate(tmpl_xyzs, axis=0)
            tmpl_depth = np.concatenate(tmpl_depths, axis=0)
            tmpl_dir = torch.concat(tmpl_dirs, dim=0)
            if len(tmpl_dir.shape) == 1:
                tmpl_dir = tmpl_dir.view(-1, 3)
            tmpl_ignore = np.concatenate(tmpl_ignore, axis=0)
            ic(tmpl_xyz.shape, tmpl_depth.shape, tmpl_dir.shape, tmpl_ignore.shape)

            tmpl_xyz = torch.from_numpy(tmpl_xyz).cuda()

        ray_batch_size = 20000

        position_dict = (torch.ones((n_grid, n_grid, n_grid, 3)) * -2).cuda()
        direction_dict = (torch.ones((n_grid, n_grid, n_grid, 3)) * -2).cuda()
        id_dict = (torch.ones((n_grid, n_grid, n_grid)) * -2).cuda().long()

        for i in tqdm(range((len(tmpl_xyz)) // ray_batch_size + (int)((len(tmpl_xyz)) % ray_batch_size != 0))):
            xyz_batch = tmpl_xyz[i * ray_batch_size:i * ray_batch_size + ray_batch_size]
            ref_dir_batch = tmpl_dir[i * ray_batch_size:i * ray_batch_size + ray_batch_size]
            id_batch = torch.tensor(
                [m for m in range(i * ray_batch_size, min((i + 1) * ray_batch_size, len(tmpl_xyz)))]).long().cuda()
            index_int = (xyz_batch * n_grid).long()
            index_int = torch.where(index_int > n_grid - 1, n_grid - 1, index_int)
            index_int = torch.where(index_int < 0, 0, index_int)
            position_dict[index_int[:, 0], index_int[:, 1], index_int[:, 2]] = xyz_batch
            direction_dict[index_int[:, 0], index_int[:, 1], index_int[:, 2]] = ref_dir_batch.cuda()
            id_dict[index_int[:, 0], index_int[:, 1], index_int[:, 2]] = id_batch

        torch.cuda.empty_cache()

        # Step C. Next, in the training rays, find the top K values (currently, k=3)
        # or index values using nearest neighbor search, and use them as the corresponding ray results
        # (Verification: A simple verification can be done using rendering and pixel replacement).

        res_dict = (torch.ones((len(cur_time_idx_list), W * H, 3)) * -1).cuda()
        for test_idx in tqdm(range(len(cur_time_idx_list))):
            data = train_dataset[cur_time_idx_list[test_idx]]
            samples, gt_rgb, sample_times = data["rays"], data["rgbs"], data["time"]
            rays = samples.view(-1, samples.shape[-1])
            times = sample_times.view(-1, sample_times.shape[-1])

            ref_xyz = torch.from_numpy(xyzs[test_idx].reshape(-1, 3)).cuda()
            ref_dir = rays[..., 3:].cuda()

            style_tmpl = torch.from_numpy(style_imgs[tmpl_id].reshape(-1, 3) ).cuda()
            ray_batch_size = 20000

            for i in range((H * W) // ray_batch_size + (int)((H * W) % ray_batch_size != 0)):
                xyz_batch = ref_xyz[i * ray_batch_size:i * ray_batch_size + ray_batch_size]
                ref_dir_batch = ref_dir[i * ray_batch_size:i * ray_batch_size + ray_batch_size]

                index_int = (xyz_batch * n_grid).long()
                grid_pos = position_dict[index_int[:, 0], index_int[:, 1], index_int[:, 2]]
                grid_dir = direction_dict[index_int[:, 0], index_int[:, 1], index_int[:, 2]]
                grid_idx = id_dict[index_int[:, 0], index_int[:, 1], index_int[:, 2]]
                cur_dist = torch.sum((xyz_batch - grid_pos) ** 2, dim=1)
                # mask with a direction distance restriction.
                dir_dist = torch.nn.functional.cosine_similarity(ref_dir_batch, grid_dir, dim=1)
                cur_dist[dir_dist < 0.3] = 100

                res_dict[test_idx][i * ray_batch_size:i * ray_batch_size + ray_batch_size] = style_tmpl[grid_idx]
                res_dict[test_idx][i * ray_batch_size:i * ray_batch_size + ray_batch_size][cur_dist > 0.1] = -1

        # Save the dictionary.
        torch.save(res_dict.cpu(), os.path.join(dict_dir, f'color_corr_{tmpl_id}.pt'))
        # DONE
        # clear memory
        res_dict = res_dict.cpu()
        position_dict = position_dict.cpu()
        direction_dict = direction_dict.cpu()
        id_dict = id_dict.cpu()
        tmpl_xyz = tmpl_xyz.cpu()
        style_tmpl = style_tmpl.cpu()
        torch.cuda.empty_cache()
