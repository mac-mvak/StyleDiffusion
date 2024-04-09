import time
from glob import glob
from tqdm import tqdm
import os
import numpy as np
import cv2
from PIL import Image
import torch
from torch import nn
import torch.nn.functional as F
import torchvision.utils as tvu

from models.ddpm.diffusion import DDPM
from models.improved_ddpm.script_util import i_DDPM
from utils.text_dic import SRC_TRG_TXT_DIC
from utils.diffusion_utils import get_beta_schedule, denoising_step
from losses import id_loss
from collections import defaultdict
from losses.clip_loss import CLIPLoss
from datasets.data_utils import get_dataset, get_dataloader
from configs.paths_config import DATASET_PATHS, MODEL_PATHS, HYBRID_MODEL_PATHS, HYBRID_CONFIG
from datasets.GENERIC_dataset import GENERIC_dataset
from datasets.imagenet_dic import IMAGENET_DIC
from utils.align_utils import run_alignment

class StyleTransfer(object):
    def __init__(self, args, config, device=None):
        self.args = args
        self.config = config
        if device is None:
            device = torch.device(
                "cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.device = device

        self.model_var_type = config.model.var_type
        betas = get_beta_schedule(
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps
        )
        self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]

        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])
        posterior_variance = betas * \
                             (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        if self.model_var_type == "fixedlarge":
            self.logvar = np.log(np.append(posterior_variance[1], betas[1:]))

        elif self.model_var_type == 'fixedsmall':
            self.logvar = np.log(np.maximum(posterior_variance, 1e-20))


    def transfer_style(self):
        #print(f'   {self.src_txts}')
        #print(f'-> {self.trg_txts}')
        
        model = i_DDPM(self.config.data.dataset, self.args.image_size)
        if self.args.image_size == 256:
            model_path = 'pretrained/256x256_diffusion_uncond.pt'
        elif self.args.image_size == 512:
            model_path = 'pretrained/512x512_diffusion.pt'
        init_ckpt = torch.load(model_path)
        u = model.load_state_dict(init_ckpt)
        model.to(self.device)
        model = model

        # ----------- Precompute Latents -----------#
        print("Prepare identity latent")
        seq_inv = np.linspace(0, 1, self.args.n_inv_step) * self.args.t_0_transfer
        seq_inv = [int(s) for s in list(seq_inv)]
        seq_inv_next = [-1] + list(seq_inv[:-1])

        n = self.args.bs_train
        img_lat_pairs_dic = {}

        lats_dict = defaultdict(list)
        for mode in ['train', 'test', 'style']:
            img_lat_pairs = []
            lats = torch.load('precomputed/' #
                                          f'{self.config.data.category}_{mode}_t{self.args.t_0_remove}_nim{self.args.n_precomp_img}_ninv{self.args.n_inv_step}_pairs.pth')

            for step, xs in enumerate(lats):
                if mode == 'style':
                    img = xs[2]
                    style_color = xs[0].clone().to(self.config.device).unsqueeze(0)
                    style_gray = xs[2].clone().to(self.config.device)
                else:
                    img = xs[1]
                x0 = img.to(self.config.device)

                x = x0.clone()
                model.eval()
                time_s = time.time()
                with torch.no_grad():
                    with tqdm(total=len(seq_inv), desc=f"Inversion transfer process {mode} {step}") as progress_bar:
                        for it, (i, j) in enumerate(zip((seq_inv_next[1:]), (seq_inv[1:]))):
                            t = (torch.ones(n) * i).to(self.device)
                            t_prev = (torch.ones(n) * j).to(self.device)

                            x = denoising_step(x, t=t, t_next=t_prev, models=model,
                                                logvars=self.logvar,
                                                sampling_type='ddim',
                                                b=self.betas,
                                                eta=0,
                                                learn_sigma=True)

                            progress_bar.update(1)
                    time_e = time.time()
                    print(f'{time_e - time_s} seconds')
                    x_lat = x.clone()
                    lats_dict[mode].append((x0, x_lat))
                if step == self.args.n_precomp_img - 1:
                    break

        # ----------- Optimizer and Scheduler -----------#
        print(f"Setting optimizer with lr={self.args.lr_clip_finetune}")
        optim_ft = torch.optim.Adam(model.parameters(), weight_decay=0, lr=self.args.lr_clip_finetune)
        # optim_ft = torch.optim.SGD(model.parameters(), weight_decay=0, lr=self.args.lr_clip_finetune)#, momentum=0.9)
        init_opt_ckpt = optim_ft.state_dict()
        scheduler_ft = torch.optim.lr_scheduler.StepLR(optim_ft, step_size=1, gamma=self.args.sch_gamma)
        init_sch_ckpt = scheduler_ft.state_dict()

        # ----------- Loss -----------#
        print("Loading losses")
        clip_loss_func = CLIPLoss(
            self.device,
            lambda_direction=self.args.dir_loss,
            lambda_l1=self.args.l1_loss_w,
            clip_model=self.args.clip_model_name)

        # ----------- Finetune Diffusion Models -----------#
        print("Start finetuning")
        print(f"Sampling type: {self.args.sample_type.upper()} with eta {self.args.eta}")
        if self.args.n_train_step != 0:
            seq_train = np.linspace(0, 1, self.args.n_train_step) * self.args.t_0_transfer
            seq_train = [int(s) for s in list(seq_train)]
            print('Uniform skip type')
        else:
            seq_train = list(range(self.args.t_0_transfer))
            print('No skip')
        seq_train_next = [-1] + list(seq_train[:-1])

        seq_test = np.linspace(0, 1, self.args.n_test_step) * self.args.t_0_transfer
        seq_test = [int(s) for s in list(seq_test)]
        seq_test_next = [-1] + list(seq_test[:-1])


        model.load_state_dict(init_ckpt)
        optim_ft.load_state_dict(init_opt_ckpt)
        scheduler_ft.load_state_dict(init_sch_ckpt)

        # ----------- Train -----------#
        for it_out in range(self.args.n_iter):
            exp_id = os.path.split(self.args.exp)[-1]
            save_name = f'checkpoint/{exp_id}_-{it_out}.pth'
            if self.args.do_train:
                
                for k in range(self.args.k_r):
                    with tqdm(total=len(seq_train), desc=f"Style reconstruction iteration_{k}") as progress_bar:
                        x = lats_dict['style'][0][1].clone()
                        for t_it, (i, j) in enumerate(zip(reversed(seq_train), reversed(seq_train_next))):
                            t = (torch.ones(n) * i).to(self.device)
                            t_next = (torch.ones(n) * j).to(self.device)

                            x, x0_t = denoising_step(x, t=t, t_next=t_next, models=model,
                                                        logvars=self.logvar,
                                                        sampling_type=self.args.sample_type,
                                                        b=self.betas,
                                                        eta=self.args.eta,
                                                        learn_sigma=True,
                                                        out_x0_t=True)

                            progress_bar.update(1)
                            x = x.detach().clone()
                            loss_style = self.args.style_loss_w * F.mse_loss(style_color, x0_t)
            
                            loss_style.backward()

                            optim_ft.step()
                            optim_ft.zero_grad()


                for step, xs in enumerate(lats_dict['train']):
                    model.train()
                    time_in_start = time.time()

                    optim_ft.zero_grad()
                    x0, x_lat = xs
                    x = x_lat.clone().to(self.device)
                    x0 = x0.to(self.device)
                    with tqdm(total=len(seq_train), desc=f"CLIP iteration") as progress_bar:
                        for t_it, (i, j) in enumerate(zip(reversed(seq_train), reversed(seq_train_next))):
                            t = (torch.ones(n) * i).to(self.device)
                            t_next = (torch.ones(n) * j).to(self.device)

                            x, x0_t = denoising_step(x, t=t, t_next=t_next, models=model,
                                                        logvars=self.logvar,
                                                        sampling_type=self.args.sample_type,
                                                        b=self.betas,
                                                        eta=self.args.eta,
                                                        learn_sigma=True,
                                                        out_x0_t=True)

                            progress_bar.update(1)
                            x = x.detach().clone()

                            loss_clip = clip_loss_func(x0_t, x0, style_gray, style_color)
                            
            
                            loss_clip.backward()

                            optim_ft.step()
                            optim_ft.zero_grad()

                            #print(f"CLIP {step}-{it_out}: loss_clip: {loss_clip:.3f}")
                            # break

                    if self.args.save_train_image:
                        tvu.save_image((x0_t + 1) * 0.5, os.path.join(self.args.image_folder,
                                                                        f'train_{step}_2_clip_{it_out}_ngen{self.args.n_train_step}.png'))
                    time_in_end = time.time()
                    print(f"Training for 1 image takes {time_in_end - time_in_start:.4f}s")
                    if step == self.args.n_train_img - 1:
                        break

                
                torch.save(model.state_dict(), save_name)
                print(f'Model {save_name} is saved.')
                scheduler_ft.step()

            # ----------- Eval -----------#
            if self.args.do_test:
                if not self.args.do_train:
                    print(save_name)
                    model.module.load_state_dict(torch.load(save_name))

                model.eval()
                for step, xs in enumerate(lats_dict['test']):
                    with torch.no_grad():
                        x0, x_lat = xs
                        x = x_lat.clone().to(self.device)
                        x0 = x0.to(self.device)
                        with tqdm(total=len(seq_test), desc=f"Eval iteration") as progress_bar:
                            for i, j in zip(reversed(seq_test), reversed(seq_test_next)):
                                t = (torch.ones(n) * i).to(self.device)
                                t_next = (torch.ones(n) * j).to(self.device)

                                x = denoising_step(x, t=t, t_next=t_next, models=model,
                                                    logvars=self.logvar,
                                                    sampling_type=self.args.sample_type,
                                                    b=self.betas,
                                                    eta=self.args.eta,
                                                    learn_sigma=True)

                                progress_bar.update(1)

                        print(f"Eval {step}-{it_out}")
                        tvu.save_image((x + 1) * 0.5, os.path.join(self.args.image_folder,
                                                                    f'test_{step}_2_clip_{it_out}_ngen{self.args.n_test_step}.png'))
                        if step == self.args.n_test_img - 1:
                            break
