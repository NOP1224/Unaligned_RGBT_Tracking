class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = '/20TB/yanchengzhi/jinjiandong/Tracking/PMATrack'    # Base directory for saving network checkpoints.
        self.tensorboard_dir = '/20TB/yanchengzhi/jinjiandong/Tracking/PMATrack/tensorboard'    # Directory for tensorboard files.
        self.pretrained_networks = '/20TB/yanchengzhi/jinjiandong/Tracking/PMATrack/pretrained_networks'
        self.got10k_val_dir = ''
        self.lasot_lmdb_dir = ''
        self.got10k_lmdb_dir = ''
        self.trackingnet_lmdb_dir = ''
        self.coco_lmdb_dir = ''
        self.coco_dir = ''
        self.lasot_dir = ''
        self.got10k_dir = ''
        self.trackingnet_dir = ''
        self.depthtrack_dir = ''
        
        self.lasher_unaligned_dir = "/20TB/fenghao/data/LasHeR_Unalined/LasHeR_Unaligned/"
        self.luart_dir = '/data1/Datasets/Tracking/LUART/sequence'
        self.use_lmdb = False
        self.luart_lmdb_dir = '/data1/Datasets/Tracking/LUART/luart_train_set_withalign_sharded/'
        self.synmu_train_dir = '/20TB/dataset/SynMU-Train/'
        self.muart244_dir = '/20TB/dataset/MUART244/sequence'
