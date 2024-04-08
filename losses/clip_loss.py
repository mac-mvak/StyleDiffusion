import torch
import torchvision.transforms as transforms
import numpy as np

import clip
from PIL import Image

from utils.text_templates import imagenet_templates, part_templates, imagenet_templates_small


class DirectionLoss(torch.nn.Module):

    def __init__(self, loss_type='mse'):
        super(DirectionLoss, self).__init__()

        self.loss_type = loss_type

        self.loss_func = {
            'mse':    torch.nn.MSELoss,
            'cosine': torch.nn.CosineSimilarity,
            'mae':    torch.nn.L1Loss
        }[loss_type]()

    def forward(self, x, y):
        if self.loss_type == "cosine":
            return 1. - self.loss_func(x, y)
        
        return self.loss_func(x, y)

class CLIPLoss(torch.nn.Module):
    def __init__(self, device, lambda_direction=1., lambda_l1=10., clip_model='ViT-B/32'):
        super(CLIPLoss, self).__init__()

        self.device = device
        self.model, clip_preprocess = clip.load(clip_model, device=self.device)

        self.clip_preprocess = clip_preprocess
        
        self.preprocess = transforms.Compose([transforms.Normalize(mean=[-1.0, -1.0, -1.0], std=[2.0, 2.0, 2.0])] + # Un-normalize from [-1.0, 1.0] (GAN output) to [0, 1].
                                              clip_preprocess.transforms[:2] +                                      # to match CLIP input scale assumptions
                                              clip_preprocess.transforms[4:])                                       # + skip convert PIL to tensor

        self.target_direction      = None
        self.patch_text_directions = None

        self.direction_loss = DirectionLoss("cosine")
        self.l1_loss = torch.nn.L1Loss()
        self.style_d = None

        self.lambda_direction = lambda_direction
        self.lambda_l1 = lambda_l1



        self.model_cnn, preprocess_cnn = clip.load("RN50", device=self.device)
        self.preprocess_cnn = transforms.Compose([transforms.Normalize(mean=[-1.0, -1.0, -1.0], std=[2.0, 2.0, 2.0])] + # Un-normalize from [-1.0, 1.0] (GAN output) to [0, 1].
                                        preprocess_cnn.transforms[:2] +                                                 # to match CLIP input scale assumptions
                                        preprocess_cnn.transforms[4:])                                                  # + skip convert PIL to tensor



    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        images = self.preprocess(images).to(self.device)
        return self.model.encode_image(images)

    def encode_images_with_cnn(self, images: torch.Tensor) -> torch.Tensor:
        images = self.preprocess_cnn(images).to(self.device)
        return self.model_cnn.encode_image(images)
    
    def distance_with_templates(self, img: torch.Tensor, class_str: str, templates=imagenet_templates) -> torch.Tensor:

        text_features  = self.get_text_features(class_str, templates)
        image_features = self.get_image_features(img)

        similarity = image_features @ text_features.T

        return 1. - similarity


    def get_image_features(self, img: torch.Tensor, norm: bool = False) -> torch.Tensor:
        image_features = self.encode_images(img)
        
        if norm:
            image_features /= image_features.clone().norm(dim=-1, keepdim=True)

        return image_features

    def compute_image_direction(self, source_image: torch.Tensor, target_image: torch.Tensor) -> torch.Tensor:
        source_features = self.get_image_features(source_image)
        target_features = self.get_image_features(target_image)

        image_direction = (target_features - source_features)

        return image_direction


    def forward(self, color_content: torch.Tensor, gray_content: torch.Tensor, gray_style: torch.Tensor, color_style: torch.Tensor):
        
        if self.style_d is None:
            self.style_d = self.compute_image_direction(color_style, gray_style).detach()


        content_d = self.compute_image_direction(color_content, gray_content)

        clip_loss = 0.0

        
        if self.lambda_l1:
            clip_loss = clip_loss + self.lambda_l1 * self.l1_loss(content_d, self.style_d)

        if self.lambda_direction:
            clip_loss = clip_loss + self.lambda_direction * self.direction_loss(content_d, self.style_d)

        return clip_loss
