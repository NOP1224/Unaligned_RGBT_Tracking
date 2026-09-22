import os
# loss function related
from lib.utils.box_ops import giou_loss
from torch.nn.functional import l1_loss
from torch.nn import BCEWithLogitsLoss
# train pipeline related
from lib.train.trainers import LTRTrainer
# distributed training related
from torch.nn.parallel import DistributedDataParallel as DDP
# some more advanced functions
from lib.train.base_functions import *
# network related
from lib.models.ostrack import build_ostrack
# forward propagation related
from lib.train.actors import OSTrack_Actor
# for import modules
import importlib

from lib.utils.focal_loss import FocalLoss
from lib.utils.misc import is_main_process, is_dist_avail_and_initialized
import warnings

warnings.filterwarnings("ignore", category=UserWarning, message="An output with one or more elements was resized.*")

def slt_collate(batch):
    ret = {}
    for k in batch[0].keys():
        here_list = []
        for ex in batch:
            here_list.append(ex[k])
        ret[k] = here_list
    return ret


class SLTLoader(torch.utils.data.dataloader.DataLoader):
    """
    Data loader. Combines a dataset and a sampler, and provides
    single- or multi-process iterators over the dataset.
    """

    __initialized = False

    def __init__(self, name, dataset, training=True, batch_size=1, shuffle=False, sampler=None, batch_sampler=None,
                 num_workers=0, epoch_interval=1, collate_fn=None, stack_dim=0, pin_memory=False, drop_last=False,
                 timeout=0, worker_init_fn=None):

        if collate_fn is None:
            collate_fn = slt_collate

        super(SLTLoader, self).__init__(dataset, batch_size, shuffle, sampler, batch_sampler,
                 num_workers, collate_fn, pin_memory, drop_last,
                 timeout, worker_init_fn)

        self.name = name
        self.training = training
        self.epoch_interval = epoch_interval
        self.stack_dim = stack_dim

def run(settings):
    settings.description = 'Training script for bat'

    # update the default configs with config file
    if not os.path.exists(settings.cfg_file):
        raise ValueError("%s doesn't exist." % settings.cfg_file)
    config_module = importlib.import_module("lib.config.%s.config" % settings.script_name)
    cfg = config_module.cfg
    config_module.update_config_from_file(settings.cfg_file)
    if is_main_process():
        print("New configuration is shown below.")
        for key in cfg.keys():
            print("%s configuration:" % key, cfg[key])
            print('\n')

    # update settings based on cfg
    update_settings(settings, cfg)

    # Record the training log
    log_dir = os.path.join(settings.save_dir, 'logs')
    if is_main_process():
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
    settings.log_file = os.path.join(log_dir, "%s-%s.log" % (settings.script_name, settings.config_name))
    if is_dist_avail_and_initialized():
        torch.distributed.barrier()

    # Build dataloaders
    loader_train, loader_val = build_dataloaders(cfg, settings)

    if settings.script_name != "ostrack":
        raise ValueError("PMRL provides only the 'ostrack' training entry")
    net = build_ostrack(cfg)

    # Move to the process-local device before wrapping with DDP.
    if settings.distributed:
        settings.device = torch.device("cuda:%d" % settings.local_rank)
    else:
        settings.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = net.to(settings.device)
    if settings.distributed:
        net = DDP(
            net,
            device_ids=[settings.local_rank],
            output_device=settings.local_rank,
            find_unused_parameters=False,
            broadcast_buffers=False,
        )
    settings.deep_sup = getattr(cfg.TRAIN, "DEEP_SUPERVISION", False)
    settings.distill = getattr(cfg.TRAIN, "DISTILL", False)
    settings.distill_loss_type = getattr(cfg.TRAIN, "DISTILL_LOSS_TYPE", "KL")
    # Loss functions and Actors
    focal_loss = FocalLoss()
    objective = {'giou': giou_loss, 'l1': l1_loss, 'focal': focal_loss, 'cls': BCEWithLogitsLoss()}
    loss_weight = {'giou': cfg.TRAIN.GIOU_WEIGHT, 'l1': cfg.TRAIN.L1_WEIGHT, 'focal': 1., 'cls': 1.0}
    actor = OSTrack_Actor(net=net, objective=objective, loss_weight=loss_weight, settings=settings, cfg=cfg)

    # Optimizer, parameters, and learning rates
    optimizer, lr_scheduler = get_optimizer_scheduler(net, cfg)
    use_amp = getattr(cfg.TRAIN, "AMP", False)
    settings.save_epoch_interval = getattr(cfg.TRAIN, "SAVE_EPOCH_INTERVAL", 1)
    settings.save_last_n_epoch = getattr(cfg.TRAIN, "SAVE_LAST_N_EPOCH", 1)
    settings.save_start_epoch = getattr(cfg.TRAIN, "SAVE_START_EPOCH", 1)

    if loader_val is None:
        trainer = LTRTrainer(actor, [loader_train], optimizer, settings, lr_scheduler, use_amp=use_amp)
    else:
        trainer = LTRTrainer(actor, [loader_train, loader_val], optimizer, settings, lr_scheduler, use_amp=use_amp)
    
    # train process
    trainer.train(cfg.TRAIN.EPOCH, load_latest=True, fail_safe=not settings.distributed)
