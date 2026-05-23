class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = '/20TB/fenghao/Projects/SFCATrack'    # Base directory for saving network checkpoints.
        self.tensorboard_dir = '/20TB/fenghao/Projects/SFCATrack/tensorboard'    # Directory for tensorboard files.
        self.pretrained_networks = '/20TB/fenghao/Projects/SFCATrack/pretrained_networks'
        self.got10k_val_dir = '/20TB/fenghao/Projects/SFCATrack/data/got10k/val'
        self.lasot_lmdb_dir = '/20TB/fenghao/Projects/SFCATrack/data/lasot_lmdb'
        self.got10k_lmdb_dir = '/20TB/fenghao/Projects/SFCATrack/data/got10k_lmdb'
        self.trackingnet_lmdb_dir = '/20TB/fenghao/Projects/SFCATrack/data/trackingnet_lmdb'
        self.coco_lmdb_dir = '/20TB/fenghao/Projects/SFCATrack/data/coco_lmdb'
        self.coco_dir = '/20TB/fenghao/Projects/SFCATrack/data/coco'
        self.lasot_dir = '/20TB/fenghao/Projects/SFCATrack/data/lasot'
        self.got10k_dir = '/20TB/fenghao/Projects/SFCATrack/data/got10k/train'
        self.trackingnet_dir = '/20TB/fenghao/Projects/SFCATrack/data/trackingnet'
        self.depthtrack_dir = '/20TB/fenghao/Projects/SFCATrack/data/depthtrack/train'
        self.lasher_dir = '/20TB/yanchengzhi/data/LasHeR/trainingset'
        self.visevent_dir = '/20TB/fenghao/Projects/SFCATrack/data/visevent/train'
        self.luart_dir = "/20TB/dataset/LUART/sequence"
