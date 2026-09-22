import torch
from torch.utils.data.distributed import DistributedSampler
# datasets used by the paper
from lib.train.dataset import LUART_Dataset, LasHeR_Unaligned, LasHeR_Unaligned_Test
from lib.train.data import sampler, opencv_loader, processing, LTRLoader
import lib.train.data.transforms as tfm
from lib.utils.misc import is_main_process

def update_settings(settings, cfg):
    settings.print_interval = cfg.TRAIN.PRINT_INTERVAL
    settings.search_area_factor = {'template': cfg.DATA.TEMPLATE.FACTOR,
                                   'search': cfg.DATA.SEARCH.FACTOR}
    settings.output_sz = {'template': cfg.DATA.TEMPLATE.SIZE,
                          'search': cfg.DATA.SEARCH.SIZE}
    settings.center_jitter_factor = {'template': cfg.DATA.TEMPLATE.CENTER_JITTER,
                                     'search': cfg.DATA.SEARCH.CENTER_JITTER}
    settings.scale_jitter_factor = {'template': cfg.DATA.TEMPLATE.SCALE_JITTER,
                                    'search': cfg.DATA.SEARCH.SCALE_JITTER}
    settings.grad_clip_norm = cfg.TRAIN.GRAD_CLIP_NORM
    settings.print_stats = None
    settings.batchsize = cfg.TRAIN.BATCH_SIZE
    settings.scheduler_type = cfg.TRAIN.SCHEDULER.TYPE
    settings.fix_bn = getattr(cfg.TRAIN, "FIX_BN", False) # add for fixing base model bn layer
    # settings.use_lmdb = getattr(cfg.TRAIN, "USE_LMDB", False)
    # settings.luart_lmdb_dir = cfg.TRAIN.LMDB_DIR

def names2datasets(name_list: list, settings, image_loader, min_bias = 0, max_bias = 0, train_phase = None):
    assert isinstance(name_list, list)
    datasets = []
    for name in name_list:
        if name == "LUART_train":
            datasets.append(LUART_Dataset(settings.env.luart_dir, dtype='rgbrgb_notalign_1', split='train'
                                          ))
        elif name == "LUART_test":
            datasets.append(LUART_Dataset(settings.env.luart_dir, dtype='rgbrgb_notalign_1', split='test'))
        elif name == "LasHeR_Unaligned":
            datasets.append(LasHeR_Unaligned(settings.env.lasher_unaligned_dir, dtype='rgbrgb_notalign_1'))
        elif name == "LasHeR_Unaligned_Test":
            datasets.append(LasHeR_Unaligned_Test(settings.env.lasher_unaligned_dir, dtype='rgbrgb_notalign_1'))
        else:
            raise ValueError(f"Unsupported PMRL dataset: {name}")
    return datasets


def build_dataloaders(cfg, settings):
    # Data transform
    # Note: for multimodal data, ToGrayscale and Normalize need modify
    transform_joint = tfm.Transform(tfm.ToGrayscale(probability=0.05),
                                    tfm.RandomHorizontalFlip(probability=0.5))

    transform_train = tfm.Transform(tfm.ToTensorAndJitter(0.),
                                    tfm.RandomHorizontalFlip_Norm(probability=0.5),
                                    tfm.Normalize(mean=cfg.DATA.MEAN, std=cfg.DATA.STD))

    transform_val = tfm.Transform(tfm.ToTensor(),
                                  tfm.Normalize(mean=cfg.DATA.MEAN, std=cfg.DATA.STD))
    
    # The tracking pairs processing module
    output_sz = settings.output_sz
    search_area_factor = settings.search_area_factor

    data_processing_train = processing.SFCAProcessing(search_area_factor=search_area_factor,
                                                       output_sz=output_sz,
                                                       center_jitter_factor=settings.center_jitter_factor,
                                                       scale_jitter_factor=settings.scale_jitter_factor,
                                                       mode='sequence',
                                                       loader_mode='train',
                                                       transform=transform_train,
                                                       joint_transform=transform_joint,
                                                       settings=settings)

    data_processing_val = processing.SFCAProcessing(search_area_factor=search_area_factor,
                                                     output_sz=output_sz,
                                                     center_jitter_factor=settings.center_jitter_factor,
                                                     scale_jitter_factor=settings.scale_jitter_factor,
                                                     mode='sequence',
                                                     loader_mode='val',
                                                     transform=transform_val,
                                                     joint_transform=transform_joint,
                                                     settings=settings)

    # Train sampler and loader
    bias_min = getattr(cfg.TRAIN, "BIAS_MIN")
    bias_max = getattr(cfg.TRAIN, "BIAS_MAX")
    train_phase = getattr(cfg.TRAIN.PROMPT, "TYPE")
    settings.num_template = getattr(cfg.DATA.TEMPLATE, "NUMBER", 1)
    settings.num_search = getattr(cfg.DATA.SEARCH, "NUMBER", 1)
    sampler_mode = getattr(cfg.DATA, "SAMPLER_MODE", "causal")
    train_cls = getattr(cfg.TRAIN, "TRAIN_CLS", False)

    if is_main_process():
        print("sampler_mode", sampler_mode)
    dataset_train = sampler.TrackingSampler(datasets=names2datasets(cfg.DATA.TRAIN.DATASETS_NAME, settings, opencv_loader, min_bias=bias_min, max_bias=bias_max, train_phase=train_phase),
                                            p_datasets=cfg.DATA.TRAIN.DATASETS_RATIO,
                                            samples_per_epoch=cfg.DATA.TRAIN.SAMPLE_PER_EPOCH,
                                            max_gap=cfg.DATA.MAX_SAMPLE_INTERVAL, bias_min=bias_min, bias_max=bias_max, num_search_frames=settings.num_search,
                                            num_template_frames=settings.num_template, processing=data_processing_train,
                                            frame_sample_mode=sampler_mode, train_cls=train_cls)

    train_sampler = DistributedSampler(dataset_train, shuffle=True) if getattr(settings, "distributed", False) else None
    shuffle = False if getattr(settings, "distributed", False) else True

    loader_train = LTRLoader('train', dataset_train, training=True, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=shuffle,
                             num_workers=cfg.TRAIN.NUM_WORKER, drop_last=True, stack_dim=1, sampler=train_sampler)

    # Validation samplers and loaders(visevent no val split)
    if cfg.DATA.VAL.DATASETS_NAME[0] is None:
        loader_val = None
    else:
        dataset_val = sampler.TrackingSampler(datasets=names2datasets(cfg.DATA.VAL.DATASETS_NAME, settings, opencv_loader),
                                            p_datasets=cfg.DATA.VAL.DATASETS_RATIO,
                                            samples_per_epoch=cfg.DATA.VAL.SAMPLE_PER_EPOCH,
                                            max_gap=cfg.DATA.MAX_SAMPLE_INTERVAL, bias_min=bias_min, bias_max=bias_max, num_search_frames=settings.num_search,
                                            num_template_frames=settings.num_template, processing=data_processing_val,
                                            frame_sample_mode=sampler_mode, train_cls=train_cls)
        val_sampler = DistributedSampler(dataset_val, shuffle=False) if getattr(settings, "distributed", False) else None
        loader_val = LTRLoader('val', dataset_val, training=False, batch_size=cfg.TRAIN.BATCH_SIZE,
                            num_workers=cfg.TRAIN.NUM_WORKER, drop_last=True, stack_dim=1, sampler=val_sampler,
                            epoch_interval=cfg.TRAIN.VAL_EPOCH_INTERVAL)

    return loader_train, loader_val


def get_optimizer_scheduler(net, cfg):

    def _make_group(params, lr=None, weight_decay=None):
        params = [p for p in params if p.requires_grad]
        if len(params) == 0:
            return None

        group = {"params": params}

        if lr is not None:
            group["lr"] = lr

        if weight_decay is not None:
            group["weight_decay"] = weight_decay

        return group

    def _drop_empty_groups(groups):
        return [g for g in groups if g is not None and len(g["params"]) > 0]

    def _is_no_weight_decay(name, param):
        """
        Only parameters with ndim <= 1 do not use weight decay.

        This typically covers bias, normalization affine parameters,
        and scalar parameters without relying on parameter names.
        """
        return param.ndim <= 1

    def _split_decay_no_decay(named_params):
        decay_params = []
        no_decay_params = []

        for n, p in named_params:
            if not p.requires_grad:
                continue

            if _is_no_weight_decay(n, p):
                no_decay_params.append(p)
            else:
                decay_params.append(p)

        return decay_params, no_decay_params

    def build_single_stage_param_groups():
        """
        Offset-model single-stage training:
        - non-backbone: cfg.TRAIN.LR
        - alignment modules: cfg.TRAIN.LR
        - backbone: cfg.TRAIN.LR * cfg.TRAIN.BACKBONE_MULTIPLIER

        In each stage, parameters are split into decay / no_decay groups.
        """
        non_backbone_named_params = [
            (n, p) for n, p in net.named_parameters()
            if "backbone" not in n
            and "progressive_alignment" not in n
            and "tcmda" not in n
            and p.requires_grad
        ]

        alignment_named_params = [
            (n, p) for n, p in net.named_parameters()
            if ("progressive_alignment" in n or "tcmda" in n)
            and p.requires_grad
        ]

        backbone_named_params = [
            (n, p) for n, p in net.named_parameters()
            if "backbone" in n
            and "progressive_alignment" not in n
            and "tcmda" not in n
            and p.requires_grad
        ]

        non_backbone_decay, non_backbone_no_decay = _split_decay_no_decay(non_backbone_named_params)
        alignment_decay, alignment_no_decay = _split_decay_no_decay(alignment_named_params)
        backbone_decay, backbone_no_decay = _split_decay_no_decay(backbone_named_params)

        return _drop_empty_groups([
            # non-backbone params
            _make_group(
                non_backbone_decay,
                lr=cfg.TRAIN.LR,
                weight_decay=cfg.TRAIN.WEIGHT_DECAY,
            ),
            _make_group(
                non_backbone_no_decay,
                lr=cfg.TRAIN.LR,
                weight_decay=0.0,
            ),

            # progressive relation and fusion parameters
            _make_group(
                alignment_decay,
                lr=cfg.TRAIN.LR,
                weight_decay=cfg.TRAIN.WEIGHT_DECAY,
            ),
            _make_group(
                alignment_no_decay,
                lr=cfg.TRAIN.LR,
                weight_decay=0.0,
            ),

            # backbone params
            _make_group(
                backbone_decay,
                lr=cfg.TRAIN.LR * cfg.TRAIN.BACKBONE_MULTIPLIER,
                weight_decay=cfg.TRAIN.WEIGHT_DECAY,
            ),
            _make_group(
                backbone_no_decay,
                lr=cfg.TRAIN.LR * cfg.TRAIN.BACKBONE_MULTIPLIER,
                weight_decay=0.0,
            ),
        ])

    # ============================================================
    # Build param groups
    # ============================================================

    param_dicts = build_single_stage_param_groups()
    base_lr = cfg.TRAIN.LR

    # ============================================================
    # Print optimizer params
    # ============================================================

    pid2groups = {}
    for gi, g in enumerate(param_dicts):
        for p in g["params"]:
            pid2groups.setdefault(id(p), []).append(gi)

    if is_main_process():
        for n, p in net.named_parameters():
            if not p.requires_grad:
                continue
            gs = pid2groups.get(id(p))
            if not gs:
                continue

            for gi in gs:
                lr = param_dicts[gi].get("lr", base_lr)
                wd = param_dicts[gi].get("weight_decay", cfg.TRAIN.WEIGHT_DECAY)
                print(f"[group={gi} lr={lr} wd={wd}] {n} {p.numel()}")

    # ============================================================
    # Optimizer
    # ============================================================

    if cfg.TRAIN.OPTIMIZER == "ADAMW":
        optimizer = torch.optim.AdamW(
            param_dicts,
            lr=cfg.TRAIN.LR,
            weight_decay=cfg.TRAIN.WEIGHT_DECAY,
        )
    else:
        raise ValueError("Unsupported Optimizer")

    # ============================================================
    # Scheduler
    # ============================================================

    if cfg.TRAIN.SCHEDULER.TYPE == "step":
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            cfg.TRAIN.LR_DROP_EPOCH,
            gamma=cfg.TRAIN.SCHEDULER.DECAY_RATE,
        )
    elif cfg.TRAIN.SCHEDULER.TYPE == "Mstep":
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=cfg.TRAIN.SCHEDULER.MILESTONES,
            gamma=cfg.TRAIN.SCHEDULER.GAMMA,
        )
    else:
        raise ValueError("Unsupported scheduler")

    return optimizer, lr_scheduler
