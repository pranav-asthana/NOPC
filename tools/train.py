import argparse
import os
import sys
from os import mkdir
from apex import amp
import shutil
import torch.nn.functional as F

sys.path.insert(0, '..')
sys.path.insert(0, '.')

from config import cfg
from data import make_data_loader, make_data_loader_custom
from engine.trainer import do_train
from modeling import build_model
from solver import make_optimizer, WarmupMultiStepLR
from layers import make_loss
from utils.logger import setup_logger
from torch.utils.tensorboard import SummaryWriter
import torch
from ignite.handlers import Checkpoint
torch.cuda.set_device(0)

cfg.merge_from_file('configs/config.yml')
cfg.freeze()

output_dir = cfg.OUTPUT_DIR

loading_model = False
if os.path.exists(output_dir):
    loading_model = True

writer = SummaryWriter(log_dir=os.path.join(output_dir,'tensorboard'))
logger = setup_logger("rendering_model", output_dir, 0)
logger.info("Running with config:\n{}".format(cfg))
shutil.copyfile('configs/config.yml', '%s/config.yml'%output_dir)

# train_loader, vertex_list,dataset = make_data_loader(cfg, is_train=True)
# dataset.__getitem__(0)
train_loader, dataset = make_data_loader_custom(cfg, is_train = True)

model = build_model(cfg).cuda()
optimizer = make_optimizer(cfg, model)

if loading_model:
    to_load = {"model": model, "optimizer": optimizer}
    chkpt_epochs_list = [int(fname.replace('nr_checkpoint_', '').replace('.pt', '')) for fname in os.listdir(output_dir) if 'nr_checkpoint' in fname]
    print('DEBUG: chkpt_epochs_list: {}'.format(chkpt_epochs_list))
    if chkpt_epochs_list:
        epoch = max(chkpt_epochs_list)
        checkpoint_fp = os.path.join(output_dir, "nr_checkpoint_" + str(epoch) + ".pt")
        checkpoint = torch.load(checkpoint_fp)
        Checkpoint.load_objects(to_load=to_load, checkpoint=checkpoint)



scheduler = WarmupMultiStepLR(optimizer, cfg.SOLVER.STEPS, cfg.SOLVER.GAMMA, cfg.SOLVER.WARMUP_FACTOR,
                               cfg.SOLVER.WARMUP_ITERS, cfg.SOLVER.WARMUP_METHOD)
loss_fn = make_loss()
# model, optimizer = amp.initialize(model, optimizer, opt_level="O1")

do_train(
        cfg,
        model,
        train_loader,
        None,
        optimizer,
        scheduler,
        loss_fn,
        writer
    )