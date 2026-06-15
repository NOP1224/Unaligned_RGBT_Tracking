from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.davis_dir = ''
    settings.got10k_lmdb_path = '/data1/Datasets/Tracking/LUART/sequence/got10k_lmdb'
    settings.got10k_path = '/data1/Datasets/Tracking/LUART/sequence/got10k'
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.itb_path = '/data1/Datasets/Tracking/LUART/sequence/itb'
    settings.lasot_extension_subset_path_path = '/data1/Datasets/Tracking/LUART/sequence/lasot_extension_subset'
    settings.lasot_lmdb_path = '/data1/Datasets/Tracking/LUART/sequence/lasot_lmdb'
    settings.lasot_path = '/data1/Datasets/Tracking/LUART/sequence/lasot'
    settings.network_path = '/20TB/yanchengzhi/jinjiandong/Tracking/PMATrack/output/test/networks'    # Where tracking networks are stored.
    settings.nfs_path = '/data1/Datasets/Tracking/LUART/sequence/nfs'
    settings.otb_path = '/data1/Datasets/Tracking/LUART/sequence/otb'
    settings.prj_dir = '/20TB/yanchengzhi/jinjiandong/Tracking/PMATrack'
    settings.result_plot_path = '/20TB/yanchengzhi/jinjiandong/Tracking/PMATrack/output/test/result_plots'
    settings.results_path = '/20TB/yanchengzhi/jinjiandong/Tracking/PMATrack/output/test/tracking_results'    # Where to store tracking results
    settings.save_dir = '/20TB/yanchengzhi/jinjiandong/Tracking/PMATrack/output'
    settings.segmentation_path = '/20TB/yanchengzhi/jinjiandong/Tracking/PMATrack/output/test/segmentation_results'
    settings.tc128_path = '/data1/Datasets/Tracking/LUART/sequence/TC128'
    settings.tn_packed_results_path = ''
    settings.tnl2k_path = '/data1/Datasets/Tracking/LUART/sequence/tnl2k'
    settings.tpl_path = ''
    settings.trackingnet_path = '/data1/Datasets/Tracking/LUART/sequence/trackingnet'
    settings.uav_path = '/data1/Datasets/Tracking/LUART/sequence/uav'
    settings.vot18_path = '/data1/Datasets/Tracking/LUART/sequence/vot2018'
    settings.vot22_path = '/data1/Datasets/Tracking/LUART/sequence/vot2022'
    settings.vot_path = '/data1/Datasets/Tracking/LUART/sequence/VOT2019'
    settings.youtubevos_dir = ''

    return settings

