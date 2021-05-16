import argparse
import os
import sys
from os import mkdir
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, '..')
sys.path.insert(0, '.')
from config import cfg
from data import make_data_loader_custom
from engine.trainer import do_train
from modeling import build_model
from solver import make_optimizer, WarmupMultiStepLR
from layers import make_loss
from utils.logger import setup_logger
from data.datasets.utils import campose_to_extrinsic, read_intrinsics
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from ignite.handlers import Checkpoint
import cv2
torch.cuda.set_device(0)

model_path = sys.argv[1]
epoch = sys.argv[2] if len(sys.argv) > 2 else None
# para_file = 'nr_model_%s.pth' % epoch
para_file = 'nr_checkpoint_%s.pt' % epoch

cfg.merge_from_file(os.path.join(model_path,'config.yml'))
cfg.SOLVER.IMS_PER_BATCH = 1
cfg.freeze()

writer = SummaryWriter(log_dir=os.path.join(model_path,'tensorboard_test'))
test_loader, dataset = make_data_loader_custom(cfg)

model = build_model(cfg)
# model.load_state_dict(torch.load(os.path.join(model_path,para_file),map_location='cpu'))
optimizer = None
to_load = {"model": model}
if epoch is None:
    epoch = max([int(fname.replace('nr_checkpoint_', '').replace('.pt', '')) for fname in os.listdir(model_path) if 'nr_checkpoint' in fname])
print("Loading model from epoch " + str(epoch))
checkpoint_fp = os.path.join(model_path, "nr_checkpoint_" + str(epoch) + ".pt")
checkpoint = torch.load(checkpoint_fp)
Checkpoint.load_objects(to_load=to_load, checkpoint=checkpoint)

if not os.path.exists(os.path.join(model_path,'res_%s'%epoch)):
    os.mkdir(os.path.join(model_path,'res_%s'%epoch))
if not os.path.exists(os.path.join(model_path,'res_%s/rgb'%epoch)):
    os.mkdir(os.path.join(model_path,'res_%s/rgb'%epoch))
if not os.path.exists(os.path.join(model_path,'res_%s/alpha'%epoch)):
    os.mkdir(os.path.join(model_path,'res_%s/alpha'%epoch))
if not os.path.exists(os.path.join(model_path,'res_%s/rgba'%epoch)):
    os.mkdir(os.path.join(model_path,'res_%s/rgba'%epoch))

model.eval()
model = model.cuda()

batch_num = 0
for batch in test_loader:
    batch_num += 1
    if batch_num % 20 == 0:
        print(f"Batch {batch_num}/{len(test_loader)}")
    # in_points = batch[1].cuda()
    # K = batch[2].cuda()
    # T = batch[3].cuda()
    # near_far_max_splatting_size = batch[5]
    # num_points = batch[4]
    # point_indexes = batch[0]
    # target = batch[-1].cuda()

    target = batch[0][0].cuda()
    K = batch[1][0].cuda()
    T = batch[2][0].cuda()
    near_far_max_splatting_size = batch[3].cuda()
    label = batch[4][0]
    inds = None

    near_far_max_splatting_size = near_far_max_splatting_size.repeat(K.shape[0], 1)    
    camNum = T.size(0)
    
    res,depth,features,dir_in_world,rgb,point_features = model(target[:1, :3, :, :], K, T,
                        near_far_max_splatting_size, None)

    depth = (depth - torch.min(depth))
    depth = depth / torch.max(depth)

    for ID in range(camNum):        
        img_t = res.detach().cpu()[ID]
        mask_t = img_t[3:4,:,:]
        img = cv2.cvtColor(img_t.permute(1,2,0).numpy()*255.0,cv2.COLOR_BGR2RGB)
        mask = mask_t.permute(1,2,0).numpy()*255.0
        rgba=img*mask/255.0+(255.0-mask)
        
        cv2.imwrite(os.path.join(model_path,'res_%s/rgb/img_'%(epoch) +str(batch_num)+'_%01d.jpg'%(ID+1)),img)
        cv2.imwrite(os.path.join(model_path,'res_%s/alpha/img_'%(epoch) +str(batch_num)+'_%01d.jpg'%(ID+1)  ),mask)
        cv2.imwrite(os.path.join(model_path,'res_%s/rgba/img_'%(epoch) +str(batch_num)+'_%01d.jpg'%(ID+1)  ),rgba)
    
print('Render done.')


