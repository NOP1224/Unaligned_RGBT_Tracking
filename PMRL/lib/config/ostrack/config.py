import yaml


class edict(dict):
    """Minimal recursive EasyDict replacement used by the original project."""

    def __init__(self, mapping=None, **kwargs):
        super().__init__()
        for key, value in dict(mapping or {}, **kwargs).items():
            self[key] = edict(value) if isinstance(value, dict) else value

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    __setattr__ = dict.__setitem__

"""Configuration for the paper's final PMRL model."""
cfg = edict()

# MODEL
cfg.MODEL = edict()
cfg.MODEL.PRETRAIN_FILE = ""
cfg.MODEL.EXTRA_MERGER = False
cfg.MODEL.RETURN_INTER = False
cfg.MODEL.RETURN_STAGES = []

# Final PMRL relation-learning settings.
cfg.MODEL.JOINT_ALIGN_FUSION = edict()
cfg.MODEL.JOINT_ALIGN_FUSION.CUE_MIXER_TEMPERATURE = 0.20
cfg.MODEL.JOINT_ALIGN_FUSION.QUALITY_FLOOR = 0.10

# MODEL.BACKBONE
cfg.MODEL.BACKBONE = edict()
cfg.MODEL.BACKBONE.TYPE = "vit_base_patch16_224"
cfg.MODEL.BACKBONE.STRIDE = 16
cfg.MODEL.BACKBONE.MID_PE = False
cfg.MODEL.BACKBONE.SEP_SEG = False
cfg.MODEL.BACKBONE.CAT_MODE = 'direct'
cfg.MODEL.BACKBONE.MERGE_LAYER = 0
cfg.MODEL.BACKBONE.ADD_CLS_TOKEN = False
cfg.MODEL.BACKBONE.CLS_TOKEN_USE_MODE = 'ignore'

cfg.MODEL.BACKBONE.CE_LOC = []
cfg.MODEL.BACKBONE.CE_KEEP_RATIO = []
cfg.MODEL.BACKBONE.CE_TEMPLATE_RANGE = 'ALL'  # choose between ALL, CTR_POINT, CTR_REC, GT_BOX

# MODEL.HEAD
cfg.MODEL.HEAD = edict()
cfg.MODEL.HEAD.TYPE = "CENTER"
cfg.MODEL.HEAD.NUM_CHANNELS = 256
# TRAIN
cfg.TRAIN = edict()
cfg.TRAIN.PROMPT = edict()
cfg.TRAIN.PROMPT.TYPE = 'bat'  # bat_12
cfg.TRAIN.LR = 0.0001
cfg.TRAIN.WEIGHT_DECAY = 0.0001
cfg.TRAIN.EPOCH = 500
cfg.TRAIN.LR_DROP_EPOCH = 400
cfg.TRAIN.BATCH_SIZE = 16
cfg.TRAIN.NUM_WORKER = 8
cfg.TRAIN.OPTIMIZER = "ADAMW"
cfg.TRAIN.BACKBONE_MULTIPLIER = 0.1
cfg.TRAIN.GIOU_WEIGHT = 2.0
cfg.TRAIN.L1_WEIGHT = 5.0
cfg.TRAIN.FREEZE_LAYERS = [0, ]
cfg.TRAIN.PRINT_INTERVAL = 50
cfg.TRAIN.VAL_EPOCH_INTERVAL = 20
cfg.TRAIN.GRAD_CLIP_NORM = 0.1
cfg.TRAIN.AMP = False
cfg.TRAIN.EXPERT_INDEX = 0
cfg.TRAIN.BIAS_MIN = 0
cfg.TRAIN.BIAS_MAX = 10
## TRAIN save cfgs
cfg.TRAIN.FIX_BN = True      ######
cfg.TRAIN.SAVE_EPOCH_INTERVAL = 1 # 1 means save model each epoch
cfg.TRAIN.SAVE_LAST_N_EPOCH = 1 # besides, last n epoch model will be saved

cfg.TRAIN.CE_START_EPOCH = 20  # candidate elimination start epoch
cfg.TRAIN.CE_WARM_EPOCH = 80  # candidate elimination warm up epoch
cfg.TRAIN.DROP_PATH_RATE = 0.1  # drop path rate for ViT backbone
# Progressive relation/alignment losses
cfg.TRAIN.DECOMP_OFFSET_WEIGHT = 10.0
cfg.TRAIN.STAGE_ALIGN_WEIGHT_REBALANCED = 0.25
cfg.TRAIN.RELATION_WEIGHT = 0.02
cfg.TRAIN.RELATION_MEAN_WEIGHT = 0.03
cfg.TRAIN.RELATION_VAR_WEIGHT = 0.02
cfg.TRAIN.RELATION_MEAN_BETA = 0.5  # token-coordinate units
cfg.TRAIN.RELATION_VAR_BETA = 0.5   # squared token-coordinate units
cfg.TRAIN.CENTER_STAGE_FACTOR = 0.20
cfg.TRAIN.SCALE_STAGE_FACTOR = 0.40
cfg.TRAIN.REFINE_STAGE_FACTOR = 0.20
cfg.TRAIN.ALIGN_DEBUG_LOG = False
cfg.TRAIN.OFFSET_DIR_MIN_PX = 5.0
cfg.TRAIN.OFFSET_SCALE_DIR_TAU = 0.03
cfg.TRAIN.OFFSET_SCALE_DIR_MIN_LOG = 0.015
cfg.TRAIN.OFFSET_XY_DIR_WEIGHT = 0.35
cfg.TRAIN.OFFSET_XY_MAG_WEIGHT = 0.15
cfg.TRAIN.OFFSET_XY_RAW_WEIGHT = 0.05
cfg.TRAIN.OFFSET_SCALE_DIR_WEIGHT = 0.30
cfg.TRAIN.OFFSET_SCALE_MAG_WEIGHT = 0.10
cfg.TRAIN.OFFSET_SCALE_RAW_WEIGHT = 0.05
cfg.TRAIN.DETAIL_DIV_WEIGHT = 0.01
cfg.TRAIN.DETAIL_PEAK_WEIGHT = 0.001
# cfg.TRAIN.USE_LMDB = False
# cfg.TRAIN.LMDB_DIR = ''
# TRAIN.SCHEDULER
cfg.TRAIN.SCHEDULER = edict()
cfg.TRAIN.SCHEDULER.TYPE = "step"
cfg.TRAIN.SCHEDULER.DECAY_RATE = 0.1
# Align

# DATA
cfg.DATA = edict()
cfg.DATA.SAMPLER_MODE = "causal"  # sampling methods
cfg.DATA.MEAN = [0.485, 0.456, 0.406]
cfg.DATA.STD = [0.229, 0.224, 0.225]
cfg.DATA.MAX_SAMPLE_INTERVAL = 200
# DATA.TRAIN
cfg.DATA.TRAIN = edict()
cfg.DATA.TRAIN.DATASETS_NAME = ["LASOT", "GOT10K_vottrain"]
cfg.DATA.TRAIN.DATASETS_RATIO = [1, 1]
cfg.DATA.TRAIN.SAMPLE_PER_EPOCH = 60000
cfg.DATA.TRAIN.BIAS_RANGE_MIN = 90
cfg.DATA.TRAIN.BIAS_RANGE_MAX = 100

# DATA.VAL
cfg.DATA.VAL = edict()
cfg.DATA.VAL.DATASETS_NAME = []
cfg.DATA.VAL.DATASETS_RATIO = [1]
cfg.DATA.VAL.SAMPLE_PER_EPOCH = 10000
# DATA.SEARCH
cfg.DATA.SEARCH = edict()
cfg.DATA.SEARCH.SIZE = 320
cfg.DATA.SEARCH.FACTOR = 5.0
cfg.DATA.SEARCH.CENTER_JITTER = 4.5
cfg.DATA.SEARCH.SCALE_JITTER = 0.5
cfg.DATA.SEARCH.NUMBER = 1
# DATA.TEMPLATE
cfg.DATA.TEMPLATE = edict()
cfg.DATA.TEMPLATE.NUMBER = 1
cfg.DATA.TEMPLATE.SIZE = 128
cfg.DATA.TEMPLATE.FACTOR = 2.0
cfg.DATA.TEMPLATE.CENTER_JITTER = 0
cfg.DATA.TEMPLATE.SCALE_JITTER = 0

# TEST
cfg.TEST = edict()
cfg.TEST.TEMPLATE_FACTOR = 2.0
cfg.TEST.TEMPLATE_SIZE = 128
cfg.TEST.SEARCH_FACTOR = 5.0
cfg.TEST.SEARCH_SIZE = 320
cfg.TEST.EPOCH = 500




def _edict2dict(dest_dict, src_edict):
    if isinstance(dest_dict, dict) and isinstance(src_edict, dict):
        for k, v in src_edict.items():
            if not isinstance(v, edict):
                dest_dict[k] = v
            else:
                dest_dict[k] = {}
                _edict2dict(dest_dict[k], v)
    else:
        return


def gen_config(config_file):
    cfg_dict = {}
    _edict2dict(cfg_dict, cfg)
    with open(config_file, 'w') as f:
        yaml.dump(cfg_dict, f, default_flow_style=False)


def _update_config(base_cfg, exp_cfg):
    if isinstance(base_cfg, dict) and isinstance(exp_cfg, edict):
        for k, v in exp_cfg.items():
            if k in base_cfg:
                if not isinstance(v, dict):
                    base_cfg[k] = v
                else:
                    _update_config(base_cfg[k], v)
            else:
                raise ValueError("{} not exist in config.py".format(k))
    else:
        return


def update_config_from_file(filename, base_cfg=None):
    exp_config = None
    with open(filename) as f:
        exp_config = edict(yaml.safe_load(f))
        if base_cfg is not None:
            _update_config(base_cfg, exp_config)
        else:
            _update_config(cfg, exp_config)
