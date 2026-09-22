import os
import argparse
import importlib
import random

import cv2 as cv
import numpy as np
import torch
import torch.backends.cudnn
import torch.distributed as dist

import _init_paths
import lib.train.admin.settings as ws_settings
from lib.utils.misc import get_rank, get_world_size, is_main_process, setup_for_distributed


def init_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_training(script_name, config_name, cudnn_benchmark=True, local_rank=-1,
                 save_dir=None, base_seed=None):
    """Run one single-GPU process or one DDP worker."""
    if save_dir is None and is_main_process():
        print("save_dir is not given. Use the current directory instead.")
    cv.setNumThreads(0)
    torch.backends.cudnn.benchmark = cudnn_benchmark

    if is_main_process():
        print('script_name: {}.py  config_name: {}.yaml'.format(script_name, config_name))

    if base_seed is not None:
        # Global rank, rather than local rank, is unique in multi-node training.
        init_seeds(int(base_seed) + get_rank())

    settings = ws_settings.Settings()
    settings.script_name = script_name
    settings.config_name = config_name
    settings.project_path = 'train/{}/{}'.format(script_name, config_name)
    settings.local_rank = local_rank
    settings.rank = get_rank()
    settings.world_size = get_world_size()
    settings.distributed = settings.world_size > 1
    settings.is_main_process = is_main_process()
    settings.save_dir = os.path.abspath(save_dir or '.')
    settings.use_lmdb = False
    prj_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    settings.cfg_file = os.path.join(prj_dir, 'experiments/%s/%s.yaml' % (script_name, config_name))
    settings.use_wandb = False
    expr_module = importlib.import_module('lib.train.train_script')
    getattr(expr_module, 'run')(settings)


def main():
    parser = argparse.ArgumentParser(description='Run a training script.')
    parser.add_argument('--script', type=str, default='ostrack')
    parser.add_argument('--config', type=str, default='pmrl_lasher')
    parser.add_argument('--cudnn_benchmark', type=bool, default=True)
    parser.add_argument('--local_rank', '--local-rank', dest='local_rank', default=-1, type=int)
    parser.add_argument('--save_dir', type=str, default='')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--vis_gpus', type=str, default='')
    args = parser.parse_args()

    env_world_size = int(os.environ.get('WORLD_SIZE', '1'))
    args.local_rank = int(os.environ.get('LOCAL_RANK', args.local_rank))
    distributed = env_world_size > 1

    if distributed:
        if args.local_rank < 0:
            raise RuntimeError('DDP requires LOCAL_RANK. Launch with torchrun.')
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend='nccl', init_method='env://')
        setup_for_distributed(get_rank() == 0)
    else:
        if args.vis_gpus:
            os.environ['CUDA_VISIBLE_DEVICES'] = args.vis_gpus
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
        setup_for_distributed(True)
        args.local_rank = -1

    try:
        run_training(
            args.script,
            args.config,
            cudnn_benchmark=args.cudnn_benchmark,
            local_rank=args.local_rank,
            save_dir=args.save_dir,
            base_seed=args.seed,
        )
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
