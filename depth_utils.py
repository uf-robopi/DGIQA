import torch
import cv2
import numpy as np
from PIL import Image
from depth_anything.dpt import DepthAnything
from depth_anything.util.transform import Resize as DA_Resize, NormalizeImage, PrepareForNet
from torchvision.transforms import Compose


class DepthGenerator:
    """
    Wraps a DepthAnything model to produce a depth map tensor 
    given a raw BGR image (numpy array).
    """
    def __init__(self, encoder: str, device: torch.device):
        # load pretrained depth model once
        self.model = DepthAnything.from_pretrained(
            f'LiheYoung/depth_anything_{encoder}14'
        ).to(device).eval()

        self.device = device

        self.transform = Compose([
            DA_Resize(
                width=518,
                height=518,
                resize_target=False,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method='lower_bound',
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ])

    def __call__(self, raw_bgr):
        rgb = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB) / 255.0
        h, w = rgb.shape[:2]
        img = rgb

        transformed = self.transform({'image': rgb})
        img = transformed['image']
        img = torch.from_numpy(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            depth = self.model(img)

        depth = torch.nn.functional.interpolate(depth[None], (h, w), mode='bilinear', align_corners=False)[0,0]
        depth_map = depth.cpu().numpy()
        depth_map = ((depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-8)) * 255
        depth_map = depth_map.astype(np.uint8)
        depth_map = cv2.cvtColor(depth_map, cv2.COLOR_GRAY2RGB)
        
        return depth_map
