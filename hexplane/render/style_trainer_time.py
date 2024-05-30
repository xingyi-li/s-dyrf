import glob
import os
import math
import sys
import gc
import glob

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from icecream import ic
import random

from hexplane.render.render import OctreeRender_trilinear_fast as renderer
from hexplane.render.render import evaluation
from hexplane.render.util.Reg import TVLoss, compute_dist_loss
from hexplane.render.util.Sampling import GM_Resi, cal_n_samples
from hexplane.render.util.util import N_to_reso
from hexplane.render.util.ref_loss import NNFMLoss, match_colors_for_image_set


class SimpleSampler:
    """
    A sampler that samples a batch of ids randomly.
    """

    def __init__(self, total, batch):
        self.total = total
        self.batch = batch
        self.curr = total
        self.ids = None

    def nextids(self):
        self.curr += self.batch
        if self.curr + self.batch > self.total:
            self.ids = torch.LongTensor(np.random.permutation(self.total))
            self.curr = 0
        return self.ids[self.curr : self.curr + self.batch]


class StyleTrainer:
    def __init__(
        self,
        model,
        cfg,
        reso_cur,
        train_dataset,
        test_dataset,
        summary_writer,
        logfolder,
        device,
        style_imgs_o,
        tmpl_ids,
    ):
        self.model = model
        self.cfg = cfg
        self.reso_cur = reso_cur
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.summary_writer = summary_writer
        self.logfolder = logfolder
        self.device = device
        self.style_imgs_o = style_imgs_o
        self.tmpl_ids = tmpl_ids

    def get_lr_decay_factor(self, step):
        """
        Calculate the learning rate decay factor = current_lr / initial_lr.
        """
        if self.cfg.optim.lr_decay_step == -1:
            self.cfg.optim.lr_decay_step = self.cfg.optim.n_iters

        if self.cfg.optim.lr_decay_type == "exp":  # exponential decay
            lr_factor = self.cfg.optim.lr_decay_target_ratio ** (
                step / self.cfg.optim.lr_decay_step
            )
        elif self.cfg.optim.lr_decay_type == "linear":  # linear decay
            lr_factor = self.cfg.optim.lr_decay_target_ratio + (
                1 - self.cfg.optim.lr_decay_target_ratio
            ) * (1 - step / self.cfg.optim.lr_decay_step)
        elif self.cfg.optim.lr_decay_type == "cosine":  # consine decay
            lr_factor = self.cfg.optim.lr_decay_target_ratio + (
                1 - self.cfg.optim.lr_decay_target_ratio
            ) * 0.5 * (1 + math.cos(math.pi * step / self.cfg.optim.lr_decay_step))

        return lr_factor

    def get_voxel_upsample_list(self):
        """
        Precompute  spatial and temporal grid upsampling sizes.
        """
        upsample_list = self.cfg.model.upsample_list
        if (
            self.cfg.model.upsampling_type == "unaligned"
        ):  # logaritmic upsampling. See explation of "unaligned" in model/__init__.py.
            N_voxel_list = (
                torch.round(
                    torch.exp(
                        torch.linspace(
                            np.log(self.cfg.model.N_voxel_init),
                            np.log(self.cfg.model.N_voxel_final),
                            len(upsample_list) + 1,
                        )
                    )
                ).long()
            ).tolist()[1:]
        elif (
            self.cfg.model.upsampling_type == "aligned"
        ):  # aligned upsampling doesn't need precompute N_voxel_list.
            N_voxel_list = None
        # logaritmic upsampling for time grid.
        Time_grid_list = (
            torch.round(
                torch.exp(
                    torch.linspace(
                        np.log(self.cfg.model.time_grid_init),
                        np.log(self.cfg.model.time_grid_final),
                        len(upsample_list) + 1,
                    )
                )
            ).long()
        ).tolist()[1:]
        self.N_voxel_list = N_voxel_list
        self.Time_grid_list = Time_grid_list

    def sample_data(self, train_dataset, iteration):
        """
        Sample a batch of data from the dataset.
        """
        train_depth = None
        # sample rays: shuffle all the rays of training dataset and sampled a batch of rays from them.
        if self.cfg.data.datasampler_type == "rays":
            ray_idx = self.sampler.nextids()
            data = train_dataset[ray_idx]
            rays_train, rgb_train, frame_time = (
                data["rays"],
                data["rgbs"].to(self.device),
                data["time"],
            )
            if self.depth_data:
                train_depth = data["depths"].to(self.device)
        # sample images: randomly pick one image from the training dataset and sample a batch of rays from all the rays of the image.
        elif self.cfg.data.datasampler_type == "images":
            img_i = self.sampler.nextids()
            data = train_dataset[img_i]
            rays_train, rgb_train, frame_time = (
                data["rays"],
                data["rgbs"].to(self.device).view(-1, 3),
                data["time"],
            )
            select_inds = torch.randperm(rays_train.shape[0])[
                : self.cfg.optim.batch_size
            ]
            rays_train = rays_train[select_inds]
            rgb_train = rgb_train[select_inds]
            frame_time = frame_time[select_inds]
            if self.depth_data:
                train_depth = data["depths"].to(self.device).view(-1, 1)[select_inds]
        # hierarchical sampling from dyNeRF: hierachical sampling involves three stages of samplings.
        elif self.cfg.data.datasampler_type == "hierach":
            # Stage 1: randomly sample a single image from an arbitrary camera.
            # And sample a batch of rays from all the rays of the image based on the difference of global median and local values.
            # Stage 1 only samples key-frames, which is the frame every self.cfg.data.key_f_num frames.
            if iteration <= self.cfg.data.stage_1_iteration:
                cam_i = np.random.choice(train_dataset.all_rgbs.shape[0])
                index_i = np.random.choice(
                    train_dataset.all_rgbs.shape[1] // self.cfg.data.key_f_num
                )
                rgb_train = (
                    train_dataset.all_rgbs[cam_i, index_i * self.cfg.data.key_f_num]
                    .view(-1, 3)
                    .to(self.device)
                )
                rays_train = train_dataset.all_rays[cam_i].view(-1, 6)
                frame_time = train_dataset.all_times[
                    cam_i, index_i * self.cfg.data.key_f_num
                ]
                # Calcualte the probability of sampling each ray based on the difference of global median and local values.
                probability = GM_Resi(
                    rgb_train, self.global_mean[cam_i], self.cfg.data.stage_1_gamma
                )
                select_inds = torch.multinomial(
                    probability, self.cfg.optim.batch_size
                ).to(rays_train.device)
                rays_train = rays_train[select_inds]
                rgb_train = rgb_train[select_inds]
                frame_time = torch.ones_like(rays_train[:, 0:1]) * frame_time
            elif (
                iteration
                <= self.cfg.data.stage_2_iteration + self.cfg.data.stage_1_iteration
            ):
                # Stage 2: basically the same as stage 1, but samples all the frames instead of only key-frames.
                cam_i = np.random.choice(train_dataset.all_rgbs.shape[0])
                index_i = np.random.choice(train_dataset.all_rgbs.shape[1])
                rgb_train = (
                    train_dataset.all_rgbs[cam_i, index_i].view(-1, 3).to(self.device)
                )
                rays_train = train_dataset.all_rays[cam_i].view(-1, 6)
                frame_time = train_dataset.all_times[cam_i, index_i]
                probability = GM_Resi(
                    rgb_train, self.global_mean[cam_i], self.cfg.data.stage_2_gamma
                )
                select_inds = torch.multinomial(
                    probability, self.cfg.optim.batch_size
                ).to(rays_train.device)
                rays_train = rays_train[select_inds]
                rgb_train = rgb_train[select_inds]
                frame_time = torch.ones_like(rays_train[:, 0:1]) * frame_time
            else:
                # Stage 3: randomly sample one frame and sample a batch of rays from the sampled frame.
                # TO sample a batch of rays from this frame, we calcualate the value changes of rays compared to nearby timesteps, and sample based on the value changes.
                cam_i = np.random.choice(train_dataset.all_rgbs.shape[0])
                N_time = train_dataset.all_rgbs.shape[1]
                # Sample two adjacent time steps within a range of 25 frames.
                index_i = np.random.choice(N_time)
                index_2 = np.random.choice(
                    min(N_time, index_i + 25) - max(index_i - 25, 0)
                ) + max(index_i - 25, 0)
                rgb_train = (
                    train_dataset.all_rgbs[cam_i, index_i].view(-1, 3).to(self.device)
                )
                rgb_ref = (
                    train_dataset.all_rgbs[cam_i, index_2].view(-1, 3).to(self.device)
                )
                rays_train = train_dataset.all_rays[cam_i].view(-1, 6)
                frame_time = train_dataset.all_times[cam_i, index_i]
                # Calcualte the temporal difference between the two frames as sampling probability.
                probability = torch.clamp(
                    1 / 3 * torch.norm(rgb_train - rgb_ref, p=1, dim=-1),
                    min=self.cfg.data.stage_3_alpha,
                )
                select_inds = torch.multinomial(
                    probability, self.cfg.optim.batch_size
                ).to(rays_train.device)
                rays_train = rays_train[select_inds]
                rgb_train = rgb_train[select_inds]
                frame_time = torch.ones_like(rays_train[:, 0:1]) * frame_time
        return rays_train, rgb_train, frame_time, train_depth

    def init_sampler(self, train_dataset):
        """
        Initialize the sampler for the training dataset.
        """
        if self.cfg.data.datasampler_type == "rays":
            self.sampler = SimpleSampler(len(train_dataset), self.cfg.optim.batch_size)
        elif self.cfg.data.datasampler_type == "images":
            self.sampler = SimpleSampler(len(train_dataset), 1)
        elif self.cfg.data.datasampler_type == "hierach":
            self.global_mean = train_dataset.global_mean_rgb.to(self.device)

    def train(self, global_steps, optim_type):
        torch.cuda.empty_cache()

        # load the training and testing dataset and other settings.
        train_dataset = self.train_dataset
        test_dataset = self.test_dataset
        W, H = self.train_dataset.img_wh
        model = self.model
        self.depth_data = test_dataset.depth_data
        summary_writer = self.summary_writer
        reso_cur = self.reso_cur

        ndc_ray = train_dataset.ndc_ray  # if the rays are in NDC
        white_bg = test_dataset.white_bg  # if the background is white

        # Calculate the number of samples for each ray based on the current resolution.
        nSamples = min(
            self.cfg.model.nSamples,
            cal_n_samples(reso_cur, self.cfg.model.step_ratio),
        )

        # Filter the rays based on the bbox
        if (self.cfg.data.datasampler_type == "rays") and (ndc_ray is False):
            allrays, allrgbs, alltimes = (
                train_dataset.all_rays,
                train_dataset.all_rgbs,
                train_dataset.all_times,
            )
            if self.depth_data:
                alldepths = train_dataset.all_depths
            else:
                alldepths = None

            allrays, allrgbs, alltimes, alldepths = model.filtering_rays(
                allrays, allrgbs, alltimes, alldepths, bbox_only=True
            )
            train_dataset.all_rays = allrays
            train_dataset.all_rgbs = allrgbs
            train_dataset.all_times = alltimes
            train_dataset.all_depths = alldepths

        # initialize the data sampler
        self.init_sampler(train_dataset)
        # precompute the voxel upsample list
        self.get_voxel_upsample_list()

        # Initialiaze TV loss on planse
        tvreg_s = TVLoss()  # TV loss on the spatial planes
        tvreg_s_t = TVLoss(
            1.0, self.cfg.model.TV_t_s_ratio
        )  # TV loss on the spatial-temporal planes

        # initialize the NNFM loss function
        nnfm_loss_fn = NNFMLoss(device=self.device)

        if optim_type == 'rgb':
            pbar = tqdm(
                range(global_steps, self.cfg.optim.n_iters),
                miniters=self.cfg.systems.progress_refresh_rate,
                file=sys.stdout,
            )
        elif optim_type == 'style':
            pbar = tqdm(
                range(global_steps, self.cfg.optim.n_iters_style),
                miniters=self.cfg.systems.progress_refresh_rate,
                file=sys.stdout,
            )

            with torch.no_grad():
                tmpl_imgs = []
                for tmpl_id in self.tmpl_ids[::len(self.tmpl_ids)//3]:
                    if self.cfg.style.dataset_type == "neural3D_NDC":
                        data = test_dataset[tmpl_id]
                        samples, gt_rgb, sample_times = data["rays"], data["rgbs"], data["time"]

                        rays = samples.view(-1, samples.shape[-1])
                        times = sample_times.view(-1, sample_times.shape[-1])
                    elif self.cfg.style.dataset_type == "dnerf":
                        sample_times = torch.FloatTensor([(tmpl_id) / 119. * 2.0 - 1.0]) * test_dataset.time_scale
                        data = test_dataset[self.cfg.style.tmpl_idx_test[0]]
                        samples = data["rays"]
                        times = sample_times.expand(samples.shape[0], 1)
                        depth = None

                        rays = samples.view(-1, samples.shape[-1])
                        times = times.view(-1, times.shape[-1])

                    rgb_map, _, _, _, _ = renderer(
                        rays,
                        times,
                        model,
                        chunk=4096,
                        N_samples=-1,
                        ndc_ray=train_dataset.ndc_ray,
                        white_bg=train_dataset.white_bg,
                        device=self.device,
                    )
                    tmpl_imgs += [rgb_map.reshape(H, W, 3).permute(2, 0, 1).unsqueeze(0)]

                if self.cfg.style.style_multiview:
                    for idx in self.cfg.style.multiview_img_cam_idx:
                        if self.cfg.style.dataset_type == "neural3D_NDC":
                            data = train_dataset[(idx - 1) * train_dataset.time_number]
                            samples, gt_rgb, sample_times = data["rays"], data["rgbs"], data["time"]

                            rays = samples.view(-1, samples.shape[-1])
                            times = sample_times.view(-1, sample_times.shape[-1])
                        else:
                            raise NotImplementedError

                        rgb_map, _, _, _, _ = renderer(
                            rays,
                            times,
                            model,
                            chunk=4096,
                            N_samples=-1,
                            ndc_ray=train_dataset.ndc_ray,
                            white_bg=train_dataset.white_bg,
                            device=self.device,
                        )
                        tmpl_imgs += [rgb_map.reshape(H, W, 3).permute(2, 0, 1).unsqueeze(0)]

                    import cv2
                    style_imgs_multiview = [
                        cv2.cvtColor(cv2.imread(self.cfg.style.style_img_multiview[i], 1), cv2.COLOR_BGR2RGB).astype(
                            np.float32) / 255.0
                        for i in range(len(self.cfg.style.style_img_multiview))]
                    style_imgs_multiview = [torch.from_numpy(img).to(self.device) for img in style_imgs_multiview]

                    if self.cfg.style.fast:
                        tmpl_imgs = [F.interpolate(tmpl, size=None, scale_factor=0.5, mode="bilinear") for tmpl in tmpl_imgs]
                    nnfm_loss_fn.preload_golden_template(tmpl_imgs,
                                                         self.style_imgs_o[::len(self.tmpl_ids)//3] + style_imgs_multiview,
                                                         blocks=self.cfg.style.vgg_blocks)

                else:
                    if self.cfg.style.fast:
                        tmpl_imgs = [F.interpolate(tmpl, size=None, scale_factor=0.5, mode="bilinear") for tmpl in tmpl_imgs]
                    nnfm_loss_fn.preload_golden_template(tmpl_imgs,
                                                         self.style_imgs_o[::len(self.tmpl_ids)//3],
                                                         blocks=self.cfg.style.vgg_blocks)

        else:
            raise NotImplementedError

        PSNRs, PSNRs_test = [], [0]
        torch.cuda.empty_cache()

        # Initialize the optimizer
        grad_vars = model.get_optparam_groups(self.cfg.optim)
        optimizer = torch.optim.Adam(
            grad_vars, betas=(self.cfg.optim.beta1, self.cfg.optim.beta2)
        )

        num_dict = glob.glob(os.path.join(self.cfg.style.out_dir, "color_corr", "color_corr_*.pt"))
        related_rays_gts = []
        if self.cfg.style.dataset_type == "neural3D_NDC":
            all_dict_index = list(range(0, len(num_dict)))
            random.shuffle(all_dict_index)

            for i in range(0, len(num_dict) // 3):
                related_rays_gt = torch.load(
                    os.path.join(self.cfg.style.out_dir, "color_corr", f"color_corr_{all_dict_index[i]}.pt")).reshape(-1, 3)
                related_rays_gts.append(related_rays_gt)
        elif self.cfg.style.dataset_type == "dnerf":
            for i in range(0, len(num_dict)):
                related_rays_gt = torch.load(
                    os.path.join(self.cfg.style.out_dir, "color_corr", f"color_corr_{i}.pt")).reshape(-1, 3)
                related_rays_gts.append(related_rays_gt)

        for iteration in pbar:
            if optim_type == 'rgb':
                # Sample dat
                rays_train, rgb_train, frame_time, depth = self.sample_data(
                    train_dataset, iteration
                )
                # Render the rgb values of rays
                rgb_map, alphas_map, depth_map, weights, uncertainty = renderer(
                    rays_train,
                    frame_time,
                    model,
                    chunk=self.cfg.optim.batch_size,
                    N_samples=nSamples,
                    white_bg=white_bg,
                    ndc_ray=ndc_ray,
                    device=self.device,
                    is_train=True,
                )

                # Calculate the loss
                loss = torch.mean((rgb_map - rgb_train) ** 2)
                total_loss = loss
            elif optim_type == 'style':
                if self.cfg.style.dataset_type == "neural3D_NDC":
                    if iteration <= self.cfg.optim.n_iters + 1000:
                        cur_time_idx_list = list(range(0, len(train_dataset), 300))
                    elif iteration <= self.cfg.optim.n_iters + 2000:
                        cur_time_idx_list = list(range(0, len(train_dataset), self.cfg.data.key_f_num))
                    else:
                        cur_time_idx_list = list(range(0, len(train_dataset)))
                elif self.cfg.style.dataset_type == "dnerf":
                    cur_time_idx_list = list(range(0, len(train_dataset)))

                num_views = len(cur_time_idx_list)
                img_id = cur_time_idx_list[np.random.randint(low=0, high=num_views)]
                data = train_dataset[img_id]
                samples, gt_rgb, sample_times = data["rays"], data["rgbs"], data["time"]
                rays_train = samples.view(-1, samples.shape[-1])
                rgb_gt = gt_rgb.view(-1, gt_rgb.shape[-1])
                times_train = sample_times.view(-1, sample_times.shape[-1])

                total_loss = 0

                tmpl_id = np.random.choice(self.tmpl_ids)
                if self.cfg.style.dataset_type == "neural3D_NDC":
                    data = test_dataset[tmpl_id]
                    samples, gt_rgb, sample_times = data["rays"], data["rgbs"], data["time"]

                    rays = samples.view(-1, samples.shape[-1])
                    times = sample_times.view(-1, sample_times.shape[-1])

                    tmpl_gt_sample = self.style_imgs_o[tmpl_id].view(-1, 3).contiguous()
                    idx = np.random.randint(len(tmpl_gt_sample), size=self.cfg.optim.batch_size)
                    tmpl_gt_sample = tmpl_gt_sample[idx]
                    rays_tmpl = rays[idx]
                    times_tmpl = times[idx]
                elif self.cfg.style.dataset_type == "dnerf":
                    sample_times = torch.FloatTensor([(tmpl_id) / 119. * 2.0 - 1.0]) * test_dataset.time_scale
                    data = test_dataset[self.cfg.style.tmpl_idx_test[0]]
                    samples = data["rays"]
                    rays = samples.view(-1, samples.shape[-1])
                    times = sample_times.expand(samples.shape[0], 1)

                    tmpl_gt_sample = self.style_imgs_o[tmpl_id].view(-1, 3).contiguous()
                    idx = np.random.randint(len(tmpl_gt_sample), size=self.cfg.optim.batch_size)
                    tmpl_gt_sample = tmpl_gt_sample[idx]
                    rays_tmpl = rays[idx]
                    times_tmpl = times[idx]

                # Render the rgb values of rays
                rgb_map, _, _, _, _ = renderer(
                    rays_tmpl,
                    times_tmpl,
                    model,
                    chunk=self.cfg.optim.batch_size,
                    N_samples=nSamples,
                    white_bg=white_bg,
                    ndc_ray=ndc_ray,
                    device=self.device,
                    is_train=True,
                )

                # Calculate the loss
                loss = torch.mean((rgb_map - tmpl_gt_sample) ** 2)
                total_loss += loss

                if self.cfg.style.dataset_type == "neural3D_NDC":
                    if iteration == self.cfg.optim.n_iters + 1666:
                        del related_rays_gts[:]
                        gc.collect()
                        for i in range(len(num_dict) // 3, len(num_dict) // 3 * 2):
                            related_rays_gt = torch.load(
                                os.path.join(self.cfg.style.out_dir, "color_corr", f"color_corr_{all_dict_index[i]}.pt")).reshape(-1, 3)
                            related_rays_gts.append(related_rays_gt)
                    elif iteration == self.cfg.optim.n_iters + 1666 + 1666:
                        del related_rays_gts[:]
                        gc.collect()
                        for i in range(len(num_dict) // 3 * 2, len(num_dict)):
                            related_rays_gt = torch.load(
                                os.path.join(self.cfg.style.out_dir, "color_corr", f"color_corr_{all_dict_index[i]}.pt")).reshape(-1, 3)
                            related_rays_gts.append(related_rays_gt)

                if self.cfg.style.dataset_type == "neural3D_NDC":
                    related_rays_gt = related_rays_gts[tmpl_id % (len(num_dict) // 3)]
                elif self.cfg.style.dataset_type == "dnerf":
                    related_rays_gt = related_rays_gts[tmpl_id % len(num_dict)]

                try:
                    related_origins = train_dataset.all_rays[:, :, :3].reshape(-1, 3)[(related_rays_gt != -1).all(dim=1)]
                    related_dirs = train_dataset.all_rays[:, :, 3:].reshape(-1, 3)[
                        (related_rays_gt != -1).all(dim=1)]
                    related_rays = torch.cat([related_origins.reshape(-1, 3), related_dirs.reshape(-1, 3)], dim=1)
                except:
                    ic("error!")
                    ic("tmpl_id:", tmpl_id)
                    ic("train_dataset.all_rays.shape:", train_dataset.all_rays.shape)
                    ic("related_rays_gt.shape:", related_rays_gt.shape)
                    ic("related_rays_gt != -1:", (related_rays_gt!=-1).sum())
                    ic("related_origins.shape", related_origins.shape)
                    ic("related_dirs.shape", related_dirs.shape)
                    sys.exit()

                if self.cfg.style.dataset_type == "neural3D_NDC":
                    related_times = (train_dataset.all_times[0, tmpl_id] * torch.ones_like(train_dataset.all_rays[:, :, 0:1].reshape(-1)))[related_rays_gt.reshape(-1, 3)[:, 0] != -1]
                elif self.cfg.style.dataset_type == "dnerf":
                    related_times = ((torch.FloatTensor([(tmpl_id) / 119. * 2.0 - 1.0]) * test_dataset.time_scale) * torch.ones_like(train_dataset.all_rays[:, :, 0:1].reshape(-1)))[related_rays_gt.reshape(-1, 3)[:, 0] != -1]


                related_rays = related_rays.cuda()
                related_times = related_times.cuda()

                related_rays_gt = related_rays_gt[(related_rays_gt != -1).all(dim=1)].reshape(-1, 3).cuda()

                idx = np.random.randint(len(related_rays), size=self.cfg.optim.batch_size)
                rays_tmpl = related_rays[idx]
                times_tmpl = related_times[idx]
                tmpl_gt_sample = related_rays_gt[idx]

                rgb_map, depth_map, _, _, _ = renderer(
                    rays_tmpl,
                    times_tmpl,
                    model,
                    chunk=self.cfg.optim.batch_size,
                    N_samples=nSamples,
                    white_bg=white_bg,
                    ndc_ray=ndc_ray,
                    device=self.device,
                    is_train=True,
                )

                loss = torch.mean((rgb_map - tmpl_gt_sample) ** 2)
                total_loss += loss

                def compute_image_loss(rgb_gt):
                    with torch.no_grad():
                        rgb_pred, _, _, _, _ = renderer(
                            rays_train,
                            times_train,
                            model,
                            chunk=self.cfg.optim.batch_size,
                            N_samples=nSamples,
                            white_bg=white_bg,
                            ndc_ray=ndc_ray,
                            device=self.device,
                            is_train=False,
                        )

                        rgb_pred = rgb_pred.view(H, W, 3).permute(2, 0, 1).unsqueeze(0).contiguous()
                        rgb_gt = rgb_gt.view(H, W, 3).permute(2, 0, 1).unsqueeze(0).contiguous().to(self.device)

                    rgb_pred.requires_grad_(True)

                    w_variance = torch.mean(torch.pow(rgb_pred[:, :, :, :-1] - rgb_pred[:, :, :, 1:], 2))
                    h_variance = torch.mean(torch.pow(rgb_pred[:, :, :-1, :] - rgb_pred[:, :, 1:, :], 2))
                    img_tv_loss = 1.0 * (h_variance + w_variance) / 2.0

                    if self.cfg.style.fast:
                        loss_dict = nnfm_loss_fn(
                            outputs=F.interpolate(
                                rgb_pred,
                                size=None,
                                scale_factor=0.5,
                                mode="bilinear",
                            ),
                            styles=None,
                            blocks=self.cfg.style.vgg_blocks,
                            loss_names=self.cfg.style.loss_names,
                            contents=F.interpolate(
                                rgb_gt,
                                size=None,
                                scale_factor=0.5,
                                mode="bilinear",
                            ),
                        )
                    else:
                        loss_dict = nnfm_loss_fn(
                            outputs=rgb_pred,
                            styles=None,
                            blocks=self.cfg.style.vgg_blocks,
                            loss_names=self.cfg.style.loss_names,
                            contents=rgb_gt,
                        )

                    if "content_loss" in loss_dict:
                        loss_dict["content_loss"] *= self.cfg.style.content_weight
                    if "tcm_loss" in loss_dict:
                        loss_dict["tcm_loss"] *= self.cfg.style.tcm_weight
                    loss_dict["img_tv_loss"] = img_tv_loss

                    all_loss = sum(list(loss_dict.values()))
                    all_loss.backward()

                    rgb_pred_grad = rgb_pred.grad.squeeze(0).permute(1, 2, 0).contiguous().clone().detach().view(-1, 3)
                    return rgb_pred_grad, loss_dict
            else:
                raise ValueError('optim_type not supported.')


            # Calculate the learning rate decay factor
            if self.cfg.style.dataset_type == "neural3D_NDC":
                lr_factor = self.get_lr_decay_factor(iteration - 200000)
            elif self.cfg.style.dataset_type == "dnerf":
                lr_factor = self.get_lr_decay_factor(iteration - 10000)

            # regularization
            # TV loss on the density planes
            if self.cfg.model.TV_weight_density > 0:
                TV_weight_density = lr_factor * self.cfg.model.TV_weight_density
                loss_tv = model.TV_loss_density(tvreg_s, tvreg_s_t) * TV_weight_density
                total_loss = total_loss + loss_tv
                summary_writer.add_scalar(
                    "train/reg_tv_density",
                    loss_tv.detach().item(),
                    global_step=iteration,
                )

            # TV loss on the appearance planes
            if self.cfg.model.TV_weight_app > 0:
                TV_weight_app = lr_factor * self.cfg.model.TV_weight_app
                loss_tv = model.TV_loss_app(tvreg_s, tvreg_s_t) * TV_weight_app
                total_loss = total_loss + loss_tv
                summary_writer.add_scalar(
                    "train/reg_tv_app", loss_tv.detach().item(), global_step=iteration
                )

            # L1 loss on the density planes
            if self.cfg.model.L1_weight_density > 0:
                L1_weight_density = lr_factor * self.cfg.model.L1_weight_density
                loss_l1 = model.L1_loss_density() * L1_weight_density
                total_loss = total_loss + loss_l1
                summary_writer.add_scalar(
                    "train/reg_l1_density",
                    loss_l1.detach().item(),
                    global_step=iteration,
                )

            # L1 loss on the appearance planes
            if self.cfg.model.L1_weight_app > 0:
                L1_weight_app = lr_factor * self.cfg.model.L1_weight_app
                loss_l1 = model.L1_loss_app() * L1_weight_app
                total_loss = total_loss + loss_l1
                summary_writer.add_scalar(
                    "train/reg_l1_app", loss_l1.detach().item(), global_step=iteration
                )

            # Loss on the rendered and gt depth maps.
            if self.cfg.model.depth_loss and self.cfg.model.depth_loss_weight > 0:
                depth_loss = (depth_map.unsqueeze(-1) - depth) ** 2
                mask = depth != 0
                depth_loss = (
                    torch.mean(depth_loss[mask]) * self.cfg.model.depth_loss_weight
                )
                total_loss += depth_loss
                summary_writer.add_scalar(
                    "train/depth_loss",
                    depth_loss.detach().item(),
                    global_step=iteration,
                )

            # Dist loss from Mip360 paper.
            if self.cfg.model.dist_loss and self.cfg.model.dist_loss_weight > 0:
                svals = (weights - model.near_far[0]) / (
                    model.near_far[1] - model.near_far[0]
                )
                dist_loss = (
                    compute_dist_loss(alphas_map[..., :-1], svals)
                    * self.cfg.model.dist_weight
                )
                total_loss += dist_loss
                summary_writer.add_scalar(
                    "train/dist_loss", dist_loss.detach().item(), global_step=iteration
                )

            total_loss *= 10.
            optimizer.zero_grad()
            total_loss.backward()
            if optim_type == 'style':
                rgb_pred_grad, loss_dict = compute_image_loss(rgb_gt)
                for batch_start in range(0, H * W, self.cfg.optim.batch_size):
                    rgb_pred, alphas_map, depth_map, weights, uncertainty = renderer(
                        rays_train[batch_start:batch_start + self.cfg.optim.batch_size],
                        times_train[batch_start:batch_start + self.cfg.optim.batch_size],
                        model, chunk=self.cfg.optim.batch_size,
                        N_samples=nSamples, white_bg=white_bg, ndc_ray=ndc_ray, device=self.device, is_train=True)
                    if rgb_pred.requires_grad:
                        rgb_pred.backward(rgb_pred_grad[batch_start:batch_start + self.cfg.optim.batch_size])
            optimizer.step()

            try:
                loss = loss.detach().item()
            except:
                loss = loss
            PSNRs.append(-10.0 * np.log(loss) / np.log(10.0))
            summary_writer.add_scalar("train/PSNR", PSNRs[-1], global_step=iteration)
            summary_writer.add_scalar("train/mse", loss, global_step=iteration)

            # Print the current values of the losses.
            if iteration % self.cfg.systems.progress_refresh_rate == 0:
                pbar.set_description(
                    f"Iteration {iteration:05d}:"
                    + f" train_psnr = {float(np.mean(PSNRs)):.2f}"
                    + f" test_psnr = {float(np.mean(PSNRs_test)):.2f}"
                    + f" mse = {loss:.6f}"
                )
                PSNRs = []

            # Decay the learning rate.
            for param_group in optimizer.param_groups:
                param_group["lr"] = param_group["lr_org"] * lr_factor

            # Evaluation for every self.cfg.systems.vis_every steps.
            if (
                iteration % self.cfg.systems.vis_every == self.cfg.systems.vis_every - 1
                and self.cfg.data.N_vis != 0
            ):
                PSNRs_test = evaluation(
                    test_dataset,
                    model,
                    self.cfg,
                    f"{self.logfolder}/imgs_vis/",
                    prefix=f"{iteration:06d}_",
                    white_bg=white_bg,
                    N_samples=nSamples,
                    ndc_ray=ndc_ray,
                    device=self.device,
                    compute_extra_metrics=False,
                )
                summary_writer.add_scalar(
                    "test/psnr", np.mean(PSNRs_test), global_step=iteration
                )

                torch.cuda.synchronize()

            # Calculate the emptiness voxel.
            if iteration in self.cfg.model.update_emptymask_list:
                if (
                    reso_cur[0] * reso_cur[1] * reso_cur[2] < 256**3
                ):  # update volume resolution
                    reso_mask = reso_cur
                model.updateEmptyMask(tuple(reso_mask))

            # Upsample the volume grid.
            if iteration in self.cfg.model.upsample_list:
                if self.cfg.model.upsampling_type == "aligned":
                    reso_cur = [reso_cur[i] * 2 - 1 for i in range(len(reso_cur))]
                else:
                    N_voxel = self.N_voxel_list.pop(0)
                    reso_cur = N_to_reso(
                        N_voxel, model.aabb, self.cfg.model.nonsquare_voxel
                    )
                time_grid = self.Time_grid_list.pop(0)
                nSamples = min(
                    self.cfg.model.nSamples,
                    cal_n_samples(reso_cur, self.cfg.model.step_ratio),
                )
                model.upsample_volume_grid(reso_cur, time_grid)

                grad_vars = model.get_optparam_groups(self.cfg.optim, 1.0)
                optimizer = torch.optim.Adam(
                    grad_vars, betas=(self.cfg.optim.beta1, self.cfg.optim.beta2)
                )
