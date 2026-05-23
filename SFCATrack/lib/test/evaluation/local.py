from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.davis_dir = ''
    settings.got10k_lmdb_path = '/20TB/fenghao/Projects/SFCATrack/data/got10k_lmdb'
    settings.got10k_path = '/20TB/fenghao/Projects/SFCATrack/data/got10k'
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.itb_path = '/20TB/fenghao/Projects/SFCATrack/data/itb'
    settings.lasot_extension_subset_path_path = '/20TB/fenghao/Projects/SFCATrack/data/lasot_extension_subset'
    settings.lasot_lmdb_path = '/20TB/fenghao/Projects/SFCATrack/data/lasot_lmdb'
    settings.lasot_path = '/20TB/fenghao/Projects/SFCATrack/data/lasot'
    settings.network_path = '/20TB/fenghao/Projects/SFCATrack/output/test/networks'    # Where tracking networks are stored.
    settings.nfs_path = '/20TB/fenghao/Projects/SFCATrack/data/nfs'
    settings.otb_path = '/20TB/fenghao/Projects/SFCATrack/data/otb'
    settings.prj_dir = '/20TB/fenghao/Projects/SFCATrack'
    settings.result_plot_path = '/20TB/fenghao/Projects/SFCATrack/output/test/result_plots'
    settings.results_path = '/20TB/fenghao/Projects/SFCATrack/output/test/tracking_results'    # Where to store tracking results
    settings.save_dir = '/20TB/fenghao/Projects/SFCATrack/output'
    settings.segmentation_path = '/20TB/fenghao/Projects/SFCATrack/output/test/segmentation_results'
    settings.tc128_path = '/20TB/fenghao/Projects/SFCATrack/data/TC128'
    settings.tn_packed_results_path = ''
    settings.tnl2k_path = '/20TB/fenghao/Projects/SFCATrack/data/tnl2k'
    settings.tpl_path = ''
    settings.trackingnet_path = '/20TB/fenghao/Projects/SFCATrack/data/trackingnet'
    settings.uav_path = '/20TB/fenghao/Projects/SFCATrack/data/uav'
    settings.vot18_path = '/20TB/fenghao/Projects/SFCATrack/data/vot2018'
    settings.vot22_path = '/20TB/fenghao/Projects/SFCATrack/data/vot2022'
    settings.vot_path = '/20TB/fenghao/Projects/SFCATrack/data/VOT2019'
    settings.youtubevos_dir = ''

    return settings

