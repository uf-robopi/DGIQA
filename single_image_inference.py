import argparse
import random
import numpy as np
import torch
import cv2
from model.DGIQA import DGIQA
import albumentations as A
from albumentations.pytorch import ToTensorV2
from depth_utils import DepthGenerator

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_random_crops(img: torch.Tensor,
                     dep: torch.Tensor,
                     num_crops=20,
                     crop_size=224):
    """
    img, dep: torch.Tensor [C,H,W]
    returns two tensors [num_crops,C,crop_size,crop_size]
    """
    _, H, W = img.shape
    imgs, deps = [], []
    for _ in range(num_crops):
        x = random.randint(0, W - crop_size)
        y = random.randint(0, H - crop_size)
        imgs.append(img[:, y:y+crop_size, x:x+crop_size])
        deps.append(dep[:, y:y+crop_size, x:x+crop_size])
    return torch.stack(imgs), torch.stack(deps)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img', required=True, help='Path to input image')
    parser.add_argument('--weights', required=True, help='Path to model .pth')
    parser.add_argument('--encoder', default='vitl',choices=['vits','vitb','vitl'], help='DepthAnything encoder')
    parser.add_argument('--crops', type=int, default=10, help='Number of random patches')
    args = parser.parse_args()

    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = DGIQA().to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()

    transform = A.Compose([
        A.SmallestMaxSize(384),
        A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
        ToTensorV2()
    ], additional_targets={'depth':'image'})

    img = cv2.imread(args.img)
    if img is None:
        raise FileNotFoundError(f"could not read {args.img}")

    depth_generator = DepthGenerator(args.encoder, device)
    depth_map = depth_generator(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    aug = transform(image=img, depth=depth_map)
    img_t  = aug['image']
    dep_t  = aug['depth'] 

    img_crops, dep_crops = get_random_crops(img_t, dep_t, num_crops=args.crops, crop_size=224)
    with torch.no_grad():
        preds = model(img_crops.to(device),
                      dep_crops.to(device))
    mos = preds.mean().item()

    print(f"Predicted MOS score: {mos:.4f}")

if __name__=='__main__':
    main()
