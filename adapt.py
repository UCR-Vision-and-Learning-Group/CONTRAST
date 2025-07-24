import sys
import os
import random
import json

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data
from tqdm import tqdm

import torchvision
from torchvision import transforms
from torchmetrics import Accuracy
import timm

import distortions
from utils.utils import AverageMeter, write_logger, set_random_seed
import eata

# ──────────────────────────────────────────────────────────────────────────────
# Hyperparameters & Setup
# ──────────────────────────────────────────────────────────────────────────────
tent_lr          = 0.001
batch_size       = 128
com_weight_lr    = 0.4 #Rather than calculating second derivative for best lr, use a fixed lr for less computation.
com_weight_steps = 5
baseline_model   = 'eata'  
seed             = 1221
data_root = '/home/eegrad/fniloy/uda'
set_random_seed(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using CUDA:", device.type == "cuda")


# ──────────────────────────────────────────────────────────────────────────────
# Load pretrained ResNet‑18 models
# ──────────────────────────────────────────────────────────────────────────────
def load_resnet(checkpoint_path):
    m = timm.create_model('resnet18', pretrained=False, num_classes=100)
    ckpt = torch.load(checkpoint_path, map_location=device)
    m.load_state_dict(ckpt['model_state_dict'])
    return m.to(device)

model1 = load_resnet('best_model_clean.pth')
model2 = load_resnet('best_model_fog.pth')
model3 = load_resnet('best_model_snow.pth')
model4 = load_resnet('best_model_frost.pth')
model_list = [model1, model2, model3, model4]

# ──────────────────────────────────────────────────────────────────────────────
# Extract pretrained BatchNorm stats
# ──────────────────────────────────────────────────────────────────────────────
def get_feature_stat_auto(model):
    means, vars_ = [], []
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            means.append(m.running_mean.clone())
            vars_.append(m.running_var.clone())
    return means, vars_

pre_stats = [get_feature_stat_auto(m) for m in model_list]

# ──────────────────────────────────────────────────────────────────────────────
# Build tent‑style adaptation wrappers
# ──────────────────────────────────────────────────────────────────────────────
tented_models = []
for m in model_list:
    cfg, _p = eata.configure_model(m), eata.collect_params(m)
    opt = optim.Adam(_p[0], lr=tent_lr)
    wrapper = eata.EATA(cfg, opt, steps=1)
    tented_models.append(wrapper)

# ──────────────────────────────────────────────────────────────────────────────
# Distortion functions mapping
# ──────────────────────────────────────────────────────────────────────────────
noise_list = [
    'Gaussian Noise','Shot Noise','Impulse Noise','Defocus Blur','Glass Blur',
    'Motion Blur','Zoom Blur','Snow','Frost','Fog',
    'Brightness','Contrast','Elastic','Pixelate','JPEG'
]


d = {
    'Gaussian Noise': distortions.gaussian_noise,
    'Shot Noise':     distortions.shot_noise,
    'Impulse Noise':  distortions.impulse_noise,
    'Defocus Blur':   distortions.defocus_blur,
    'Glass Blur':     distortions.glass_blur,
    'Motion Blur':    distortions.motion_blur,
    'Zoom Blur':      distortions.zoom_blur,
    'Snow':           distortions.snow,
    'Frost':          distortions.frost,
    'Fog':            distortions.fog,
    'Brightness':     distortions.brightness,
    'Contrast':       distortions.contrast,
    'Elastic':        distortions.elastic_transform,
    'Pixelate':       distortions.pixelate,
    'JPEG':           distortions.jpeg_compression,
}

# ──────────────────────────────────────────────────────────────────────────────
# Configure/reset BatchNorm for tent updating
# ──────────────────────────────────────────────────────────────────────────────
def configure_model(m, use_source_stats=False):
    m.eval()
    m.requires_grad_(False)
    for layer in m.modules():
        if isinstance(layer, nn.BatchNorm2d):
            layer.requires_grad_(True)
            layer.track_running_stats = True
            if not use_source_stats:
                layer.train()
                layer.running_mean = torch.zeros(layer.num_features, device=device)
                layer.running_var  = torch.zeros(layer.num_features, device=device)
                layer.momentum     = 1.0
            else:
                layer.eval()
    return m

# ──────────────────────────────────────────────────────────────────────────────
# Calculation of Distance
# ──────────────────────────────────────────────────────────────────────────────
def dist(mean1, var1, mean2, var2):
    # mean1, var1, mean2, var2 are lists of Tensors
    final_sum = []
    for m1, v1, m2, v2 in zip(mean1, var1, mean2, var2):
        sums = []
        # convert running_var → std
        std1 = v1.sqrt()
        std2 = v2.sqrt()
        for mm1, std1_elem, mm2, std2_elem in zip(m1, std1, m2, std2):
            dis = (torch.log(std2_elem/std1_elem)
                   + (std1_elem**2 + (mm1 - mm2).square())/(2*std2_elem**2)
                   - 0.5)
            sums.append(dis)
        final_sum.append(torch.stack(sums).mean())
    return torch.stack(final_sum).mean()

# ──────────────────────────────────────────────────────────────────────────────
# Main adaptation & evaluation loop
# ──────────────────────────────────────────────────────────────────────────────
accuracy = Accuracy(task="multiclass", num_classes=100).to(device)
avg_accs = []
logger   = write_logger()

for noise in noise_list:
#     transform_noise = transforms.Compose([
#         transforms.Lambda(lambda img: d[noise](np.array(img), severity=5)),
#         transforms.Lambda(lambda arr: Image.fromarray(arr.astype(np.uint8))),
#         transforms.Resize((224,224)),
#         transforms.ToTensor(),
#     ])
    transform_noise = transforms.Compose([
                            transforms.Lambda(lambda x: d[noise](x, severity=5)),
                            transforms.ToTensor(), 
                            transforms.Resize((224,224))])
    
    dataset = torchvision.datasets.CIFAR100(
        root= data_root,
        train= False,
        transform= transform_noise,
        download= True
    )
    loader = data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=4
    )

    meter = AverageMeter()
    for imgs, tars in tqdm(loader, desc=noise):
        imgs = imgs.to(device).float()
        tars = tars.to(device)
        
        #compute fresh BN stats on raw images for each wrapper
        new_stats = []
        for tm in tented_models:
            tm.model = configure_model(tm.model, use_source_stats=False)
            _ = tm.model(imgs)
            new_stats.append(get_feature_stat_auto(tm.model))
        
        #restore tent defaults (eata uses same config)
        import tent
        for tm in tented_models:
            tm.model = tent.configure_model(tm.model)
        
        #compute distances & initial inverse weights
        dists = [
            dist(curr[0], curr[1], src[0], src[1])
            for curr, src in zip(new_stats, pre_stats)
        ]
        inv = torch.tensor([1 - (d / sum(dists)) for d in dists], device=device)
        weights = nn.Parameter(inv, requires_grad=True)
        opt_w   = optim.SGD([weights], lr=com_weight_lr, momentum=0.9, weight_decay=1e-4)
        
        #refine combination weights
        for _ in range(com_weight_steps):
            soft     = F.softmax(weights, dim=-1)
            combined = sum(soft[i] * tented_models[i].model(imgs) for i in range(len(model_list)))
            loss     = tent.softmax_entropy(combined).mean(0)
            loss.backward()
            opt_w.step()
            opt_w.zero_grad()
        
        #final prediction & accuracy
        with torch.no_grad():
            soft = F.softmax(weights, dim=-1)
            pred = sum(soft[i] * tented_models[i].model(imgs) for i in range(len(model_list)))
            meter.update(accuracy(pred, tars).item())
        
        #one update on best expert
        for tm in tented_models:
            tm.optimizer.zero_grad()
            
        best_idx = torch.argmax(soft).item()
        _ = tented_models[best_idx](imgs)
    
#     for tm in tented_models:
#         tm.reset()
    
    avg_accs.append(meter.avg)
    logger.write(M=noise, Accuracy=meter.avg)
    print(f"{noise:15s} → Acc = {meter.avg:.4f}")

print("All accuracies:", avg_accs)
