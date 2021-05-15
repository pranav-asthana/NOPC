# encoding: utf-8


from torch.utils import data

from .datasets import IBRStaticDataset
from .datasets import IBRDynamicDataset
from .datasets import shapenet_views_dataset
from .transforms import build_transforms
from .collate_batch import static_collate

def build_dataset(cfg, transforms, near_far_size, is_train=True, is_need_all_data = False):
    '''
    datasets = IBRStaticDataset('/data/SDisk/wmy/IBR_data/reaMa2/vertices.txt',
                   '/data/SDisk/wmy/IBR_data/reaMa2/CamPose.inf',
                   '/data/SDisk/wmy/IBR_data/reaMa2/Intrinsic.inf',
                   '/data/SDisk/wmy/IBR_data/reaMa2/images/',
                   transforms, near_far_size, is_need_all_data)
    '''
    
    datasets = IBRDynamicDataset(cfg.DATASETS.TRAIN,
                    cfg.DATASETS.MASK,
                    cfg.DATASETS.CENTER,
                   transforms, near_far_size)
    # TODO: Create a new dataset class to load our data (no need to load point clounds, just load images, K, T)
    return datasets


def make_data_loader(cfg, is_train=True, is_need_all_data = False):
    
    batch_size = cfg.SOLVER.IMS_PER_BATCH


    transforms = build_transforms(cfg, is_train)
    datasets = build_dataset(cfg, transforms, cfg.INPUT.NEAR_FAR_SIZE, is_train, is_need_all_data)

    num_workers = cfg.DATALOADER.NUM_WORKERS
    data_loader = data.DataLoader(
        datasets, collate_fn=static_collate, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )

    return data_loader, datasets.get_vertex_num().int(), datasets

def make_data_loader_custom(cfg, dataset_root):

    datasets = shapenet_views_dataset(root_dir=dataset_root, num_views=5, transform=None)
    num_workers = cfg.DATALOADER.NUM_WORKERS
    data_loader = data.DataLoader(
        datasets, shuffle=True, num_workers=num_workers
    )

    return data_loader, datasets