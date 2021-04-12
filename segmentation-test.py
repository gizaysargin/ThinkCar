# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 15:43:21 2021

@author: admin
"""
import os
os.chdir(r"C:\Users\admin\Desktop\Image-Segmentation\ThinkCar")
outputdir = r"C:\Users\admin\Desktop"
rootdir = r"C:\Users\admin\Desktop\Image-Segmentation\ThinkCar"
inputdir = r"C:\Users\admin\Desktop\ThinkCar\WindowsNoEditor\scenario_runner-0.9.10\_out"
import torch
import numpy as np
import torchvision as vis
import torchvision.models.segmentation as seg
from torchvision.datasets import Cityscapes
from torchvision import transforms as t
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torch import nn
import presets
import utils
import transforms
from collections import namedtuple
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

#%%


CityscapesClass = namedtuple('CityscapesClass', ['name', 'id', 'train_id', 'category', 'category_id',
                                                 'has_instances', 'ignore_in_eval', 'color'])

classes = [
    CityscapesClass('unlabeled', 0, 255, 'void', 0, False, True, (0, 0, 0)),
    CityscapesClass('ego vehicle', 1, 255, 'void', 0, False, True, (0, 0, 0)),
    CityscapesClass('rectification border', 2, 255, 'void', 0, False, True, (0, 0, 0)),
    CityscapesClass('out of roi', 3, 255, 'void', 0, False, True, (0, 0, 0)),
    CityscapesClass('static', 4, 255, 'void', 0, False, True, (0, 0, 0)),
    CityscapesClass('dynamic', 5, 255, 'void', 0, False, True, (111, 74, 0)),
    CityscapesClass('ground', 6, 255, 'void', 0, False, True, (81, 0, 81)),
    CityscapesClass('road', 7, 0, 'flat', 1, False, False, (128, 64, 128)),
    CityscapesClass('sidewalk', 8, 1, 'flat', 1, False, False, (244, 35, 232)),
    CityscapesClass('parking', 9, 255, 'flat', 1, False, True, (250, 170, 160)),
    CityscapesClass('rail track', 10, 255, 'flat', 1, False, True, (230, 150, 140)),
    CityscapesClass('building', 11, 2, 'construction', 2, False, False, (70, 70, 70)),
    CityscapesClass('wall', 12, 3, 'construction', 2, False, False, (102, 102, 156)),
    CityscapesClass('fence', 13, 4, 'construction', 2, False, False, (190, 153, 153)),
    CityscapesClass('guard rail', 14, 255, 'construction', 2, False, True, (180, 165, 180)),
    CityscapesClass('bridge', 15, 255, 'construction', 2, False, True, (150, 100, 100)),
    CityscapesClass('tunnel', 16, 255, 'construction', 2, False, True, (150, 120, 90)),
    CityscapesClass('pole', 17, 5, 'object', 3, False, False, (153, 153, 153)),
    CityscapesClass('polegroup', 18, 255, 'object', 3, False, True, (153, 153, 153)),
    CityscapesClass('traffic light', 19, 6, 'object', 3, False, False, (250, 170, 30)),
    CityscapesClass('traffic sign', 20, 7, 'object', 3, False, False, (220, 220, 0)),
    CityscapesClass('vegetation', 21, 8, 'nature', 4, False, False, (107, 142, 35)),
    CityscapesClass('terrain', 22, 9, 'nature', 4, False, False, (152, 251, 152)),
    CityscapesClass('sky', 23, 10, 'sky', 5, False, False, (70, 130, 180)),
    CityscapesClass('person', 24, 11, 'human', 6, True, False, (220, 20, 60)),
    CityscapesClass('rider', 25, 12, 'human', 6, True, False, (255, 0, 0)),
    CityscapesClass('car', 26, 13, 'vehicle', 7, True, False, (0, 0, 142)),
    CityscapesClass('truck', 27, 14, 'vehicle', 7, True, False, (0, 0, 70)),
    CityscapesClass('bus', 28, 15, 'vehicle', 7, True, False, (0, 60, 100)),
    CityscapesClass('caravan', 29, 255, 'vehicle', 7, True, True, (0, 0, 90)),
    CityscapesClass('trailer', 30, 255, 'vehicle', 7, True, True, (0, 0, 110)),
    CityscapesClass('train', 31, 16, 'vehicle', 7, True, False, (0, 80, 100)),
    CityscapesClass('motorcycle', 32, 17, 'vehicle', 7, True, False, (0, 0, 230)),
    CityscapesClass('bicycle', 33, 18, 'vehicle', 7, True, False, (119, 11, 32)),
    CityscapesClass('license plate', -1, -1, 'vehicle', 7, False, True, (0, 0, 142)),
]


class ToTensor(object):
    def __call__(self, target):
        target = torch.as_tensor(np.array(target), dtype=torch.int64)
        return target

def criterion(inputs, target):
    target = target.squeeze(1)
    losses = {}
    for name, x in inputs.items():
        losses[name] = nn.functional.cross_entropy(x, target, ignore_index=255)

    if len(losses) == 1:
        return losses['out']

    return losses['out'] + 0.5 * losses['aux']

def train_one_epoch(model, criterion, optimizer, data_loader, lr_scheduler, device, epoch, print_freq):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value}'))
    header = 'Epoch: [{}]'.format(epoch)
    for image, target in metric_logger.log_every(data_loader, print_freq, header):
        target = torch.LongTensor(np.array(target).astype(np.int64))
        image, target = image.to(device), target.to(device)
        output = model(image)
        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        lr_scheduler.step()
            
        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
        
def evaluate(model, data_loader, device, num_classes):
    model.eval()
    confmat = utils.ConfusionMatrix(num_classes)
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    with torch.no_grad():
        for image, target in metric_logger.log_every(data_loader, 100, header):
            target = torch.LongTensor(np.array(target).astype(np.int64))
            torch.cuda.empty_cache()
            image, target = image.to(device), target.to(device)
            output = model(image)
            torch.cuda.empty_cache()
            output = output['out']

            confmat.update(target.flatten(), output.argmax(1).flatten())

        confmat.reduce_from_all_processes()

    return confmat

def get_transform(train):
    base_size = 520
    crop_size = 480

    return presets.SegmentationPresetTrain(base_size, crop_size) if train else presets.SegmentationPresetEval(base_size)
#%%
root = "./CityScapes/"
transform = t.Compose([t.Resize(256), t.ToTensor()])
ttransform = t.Compose([t.Resize(256, 0), ToTensor()])
datasetTrain = Cityscapes(root=root, split="train", mode="fine",
                          target_type="semantic",
                          transform=transform, target_transform=ttransform)
datasetVal = Cityscapes(root=root, split="val", mode="fine",
                        target_type="semantic",
                        transform=transform, target_transform=ttransform)

#%%
valLoader = DataLoader(datasetVal, batch_size=2, shuffle=True, collate_fn=utils.collate_fn, drop_last=True)
trainLoader = DataLoader(datasetTrain, batch_size=2, shuffle=True, collate_fn=utils.collate_fn, drop_last=True)
#%%
num_epochs = 10
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = seg.deeplabv3_resnet101(num_classes=35)
model.classifier = DeepLabHead(2048, 35)
model.to(device)
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.02, momentum=0.9, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4000, gamma=0.5)

for epoch in range(1,num_epochs+1):
    train_one_epoch(model, criterion, optimizer, trainLoader, lr_scheduler, device, epoch, 100)
    confmat = evaluate(model, valLoader, device=device, num_classes=35)
    print(confmat)
    utils.save_on_master(
            {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch
            },
            os.path.join(outputdir, 'model{}.pth'.format(epoch)))
    
#%%
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = seg.deeplabv3_resnet101(num_classes=35)
model.classifier = DeepLabHead(2048, 35)
model.to(device)
mdl = torch.load("./models/model10.pth")

model.load_state_dict(mdl["model"])
    
#%%

from PIL import Image
from glob import glob

images = sorted(glob(inputdir+"/*.png"))

#%%

colors = []
for clas in classes:
    colors.append(clas[-1])

for image in images:

    im = Image.open(image).convert("RGB")
    tr = t.Compose([t.Resize(256), t.ToTensor()])
    imt = tr(im)
    imt = imt.unsqueeze(0)
    imt = imt.to(device)
    model.eval()

    out = model(imt)

    op = out["out"].detach().cpu().numpy()
    cop=op.argmax(1)
    cop = np.squeeze(cop).astype(np.uint8)
    
    mask = np.empty((im.size[1], im.size[0], 3))
    r = Image.fromarray(cop).resize(im.size,resample=Image.NEAREST)

    for i in range(im.size[1]):
        for j in range(im.size[0]):
            px = r.getpixel((j,i))
            mask[i,j,0] = colors[px-1][0]
            mask[i,j,1] = colors[px-1][1]
            mask[i,j,2] = colors[px-1][2]
    mask = mask.astype(np.uint8)
    
    m = Image.fromarray(mask)
    m.save(os.path.splitext(image)[0]+"-mask.png")
#%%

m = Image.fromarray(mask)
m.show()
#%%
im.show()





