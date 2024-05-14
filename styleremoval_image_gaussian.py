import time
from glob import glob
from tqdm import tqdm
import os
import numpy as np
import cv2
from PIL import Image
import torch
from torch import nn
import torchvision.utils as tvu

from models.ddpm.diffusion import DDPM
from models.improved_ddpm.script_util import i_DDPM
from utils.text_dic import SRC_TRG_TXT_DIC
from utils.diffusion_utils import get_beta_schedule, denoising_step
from losses import id_loss
from losses.clip_loss import CLIPLoss
from datasets.data_utils import get_dataset, get_dataloader
from configs.paths_config import DATASET_PATHS, MODEL_PATHS, HYBRID_MODEL_PATHS, HYBRID_CONFIG
from datasets.imagenet_dic import IMAGENET_DIC
from datasets.GAUSSIAN_dataset import GAUSSIAN_dataset
from utils.align_utils import run_alignment

class StyleRemovalImageGaussian(object):
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


    def remove_style(self):
        print(self.args.exp)
        #print(f'   {self.src_txts}')
        #print(f'-> {self.trg_txts}')
        

        # ----------- Precompute Latents -----------#
        print("Prepare style latent")
        seq_inv = np.linspace(0, 1, self.args.n_inv_step) * self.args.t_0_remove
        seq_inv = [int(s) for s in list(seq_inv)]
        seq_inv_next = [-1] + list(seq_inv[:-1])

        n = self.args.bs_train

        
        style_lat_pairs = []
        style_image_path = self.args.style_image
        style_color_ds = GAUSSIAN_dataset(style_image_path, color=True, img_size=self.args.image_size, gaussian_kernel=self.args.gaussian_kernel)
        style_gray_ds = GAUSSIAN_dataset(style_image_path, img_size=self.args.image_size, gaussian_kernel=self.args.gaussian_kernel)
        color_img = torch.from_numpy(style_color_ds[0][1])
        tvu.save_image((color_img + 1) * 0.5, os.path.join(self.args.image_folder,
                                                           f'style_color_rec_ninv{self.args.n_inv_step}.png'))

        x0 = torch.from_numpy(style_gray_ds[0][0])
        tvu.save_image((x0 + 1) * 0.5, os.path.join(self.args.image_folder, f'style_0_orig.png'))

        style_lat_pairs = [color_img.clone().unsqueeze(0), x0.clone().unsqueeze(0), x0.clone().unsqueeze(0)]

        image_name = self.args.style_image.split('.')[0]
        pairs_path = os.path.join('precomputed/',
                                          f'{self.config.data.category}_style_{image_name}_t{self.args.t_0_remove}_size{self.args.image_size}_nim{self.args.n_precomp_img}_ninv{self.args.n_inv_step}_{self.args.removal_mode}_pairs.pth')
        torch.save([style_lat_pairs], pairs_path)
