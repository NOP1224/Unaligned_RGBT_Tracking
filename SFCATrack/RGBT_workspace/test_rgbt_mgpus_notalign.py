import os
import cv2
import sys
from os.path import join, isdir, abspath, dirname
import numpy as np
import argparse
prj = join(dirname(__file__), '..')
if prj not in sys.path:
    sys.path.append(prj)

from lib.test.tracker.ostrack import OSTrack
from lib.test.tracker.sfcatrack import SFCATrack
import lib.test.parameter.sfcatrack as rgbt_adapter_params
import multiprocessing
import torch
from lib.train.dataset.depth_utils import get_x_frame
import time


def genConfig(seq_path, set_type):
    if set_type == 'RGBT234':
        ############################################  have to refine #############################################
        RGB_img_list = sorted([seq_path + '/visible/' + p for p in os.listdir(seq_path + '/visible') if os.path.splitext(p)[1] == '.jpg'])
        T_img_list = sorted([seq_path + '/infrared/' + p for p in os.listdir(seq_path + '/infrared') if os.path.splitext(p)[1] == '.jpg'])

        RGB_gt = np.loadtxt(seq_path + '/visible.txt', delimiter=',')
        T_gt = np.loadtxt(seq_path + '/infrared.txt', delimiter=',')

    elif set_type == 'DroneT':
            ############################################  have to refine #############################################
            RGB_img_list = sorted([seq_path + '/rgb/' + p for p in os.listdir(seq_path + '/rgb') if
                                   os.path.splitext(p)[1] == '.jpg'])
            T_img_list = sorted([seq_path + '/ir/' + p for p in os.listdir(seq_path + '/ir') if
                                 os.path.splitext(p)[1] == '.jpg'])

            RGB_gt = np.loadtxt(seq_path + '/rgb.txt', delimiter=',')
            T_gt = np.loadtxt(seq_path + '/ir.txt', delimiter=',')

    elif set_type == 'GTOT':
        ############################################  have to refine #############################################
        RGB_img_list = sorted([seq_path + '/v/' + p for p in os.listdir(seq_path + '/v') if os.path.splitext(p)[1] == '.png'])
        T_img_list = sorted([seq_path + '/i/' + p for p in os.listdir(seq_path + '/i') if os.path.splitext(p)[1] == '.png'])

        RGB_gt = np.loadtxt(seq_path + '/groundTruth_v.txt', delimiter=' ')
        T_gt = np.loadtxt(seq_path + '/groundTruth_i.txt', delimiter=' ')

        x_min = np.min(RGB_gt[:,[0,2]],axis=1)[:,None]
        y_min = np.min(RGB_gt[:,[1,3]],axis=1)[:,None]
        x_max = np.max(RGB_gt[:,[0,2]],axis=1)[:,None]
        y_max = np.max(RGB_gt[:,[1,3]],axis=1)[:,None]
        RGB_gt = np.concatenate((x_min, y_min, x_max-x_min, y_max-y_min),axis=1)

        x_min = np.min(T_gt[:,[0,2]],axis=1)[:,None]
        y_min = np.min(T_gt[:,[1,3]],axis=1)[:,None]
        x_max = np.max(T_gt[:,[0,2]],axis=1)[:,None]
        y_max = np.max(T_gt[:,[1,3]],axis=1)[:,None]
        T_gt = np.concatenate((x_min, y_min, x_max-x_min, y_max-y_min),axis=1)
    
    elif set_type == 'LasHeR':
        RGB_img_list = sorted([seq_path + '/visible/' + p for p in os.listdir(seq_path + '/visible') if p.endswith(".jpg")])
        T_img_list = sorted([seq_path + '/infrared/' + p for p in os.listdir(seq_path + '/infrared') if p.endswith(".jpg")])

        RGB_gt = np.loadtxt(seq_path + '/visible.txt', delimiter=',')
        T_gt = np.loadtxt(seq_path + '/infrared.txt', delimiter=',')

    elif 'VTUAV' in set_type:
        RGB_img_list = sorted([seq_path + '/rgb/' + p for p in os.listdir(seq_path + '/rgb') if p.endswith(".jpg")])
        T_img_list = sorted([seq_path + '/ir/' + p for p in os.listdir(seq_path + '/ir') if p.endswith(".jpg")])

        RGB_gt = np.loadtxt(seq_path + '/rgb.txt', delimiter=' ')
        T_gt = np.loadtxt(seq_path + '/ir.txt', delimiter=' ')
    elif set_type == 'LUART':
        RGB_img_list = sorted([seq_path + '/NotAlign/visible/' + p for p in os.listdir(seq_path + '/NotAlign/visible') if p.endswith(".jpg")])
        T_img_list = sorted([seq_path + '/NotAlign/infrared/' + p for p in os.listdir(seq_path + '/NotAlign/infrared') if p.endswith(".jpg")])

        RGB_gt = np.loadtxt(seq_path + '/visible.txt', delimiter=',')
        T_gt = np.loadtxt(seq_path + '/infrared.txt', delimiter=',')

    return RGB_img_list, T_img_list, RGB_gt, T_gt


def run_sequence(seq_name, seq_home, dataset_name, yaml_name, num_gpu=1, epoch=300, debug=0, script_name='adapter', test_mode = 'all'):
    seq_txt = seq_name
    save_name = '{}'.format(yaml_name)
    save_path = f'./RGBT_workspace/results/NotAlign/{dataset_name}/' + save_name +  '/' + seq_txt + '.txt'
    save_folder = f'./RGBT_workspace/results/NotAlign/{dataset_name}/' + save_name
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    if os.path.exists(save_path):
        print(f'-1 {seq_name}')
        return
    try:
        worker_name = multiprocessing.current_process().name
        worker_id = int(worker_name[worker_name.find('-') + 1:]) - 1
        gpu_id = worker_id % num_gpu
        torch.cuda.set_device(gpu_id)
    except:
        pass
    if script_name == 'sfcatrack':
        params = rgbt_adapter_params.parameters(yaml_name, epoch, debug, seq_name)
        mmtrack = SFCATrack(params, test_mode)  # "GTOT" # dataset_name
        tracker = SFCA_RGBT(tracker=mmtrack, seq_name=seq_name)

    seq_path = seq_home + '/' + seq_name
    print('——————————Process sequence: '+seq_name +'——————————————')
    RGB_img_list, T_img_list, RGB_gt, T_gt = genConfig(seq_path, dataset_name)
    if len(RGB_img_list) == len(RGB_gt):
        result = np.zeros_like(RGB_gt)
    else:
        result = np.zeros((len(RGB_img_list), 4), dtype=RGB_gt.dtype)
    result[0] = np.copy(RGB_gt[0])
    toc = 0
    for frame_idx, (rgb_path, T_path) in enumerate(zip(RGB_img_list, T_img_list)):
        tic = cv2.getTickCount()
        if frame_idx == 0:
            # initialization
            image_rgb_size, image_tir_size = get_x_frame(rgb_path, T_path, dtype=getattr(params.cfg.DATA,'XTYPE','rgbrgb_notalign_1'))
            tracker.initialize_notalign(image_rgb_size, image_tir_size, RGB_gt[0].tolist(), T_gt[0].tolist())  # xywh
        elif frame_idx > 0:
            # track
            image_rgb_size, image_tir_size = get_x_frame(rgb_path, T_path, dtype=getattr(params.cfg.DATA,'XTYPE','rgbrgb_notalign_1'))
            region, confidence = tracker.track_notalign(image_rgb_size, RGB_gt[frame_idx].tolist(), region_gt_tir=T_gt[frame_idx].tolist())  # xywh
            result[frame_idx] = np.array(region)
        toc += cv2.getTickCount() - tic
    toc /= cv2.getTickFrequency()
    if not debug:
        np.savetxt(save_path, result)
    print('{} , fps:{}'.format(seq_name, frame_idx / toc))


class SFCA_RGBT(object):
    def __init__(self, tracker, seq_name):
        self.tracker = tracker
        self.seq_name = seq_name

    def initialize(self, image, region):
        self.H, self.W, _ = image.shape
        gt_bbox_np = np.array(region).astype(np.float32)
        
        init_info = {'init_bbox': list(gt_bbox_np)}  # input must be (x,y,w,h)
        self.tracker.initialize(image, init_info)
    def initialize_notalign(self, image_rgb, image_tir, region_rgb, region_tir):
        self.H, self.W, _ = image_rgb.shape
        gt_bbox_np_rgb = np.array(region_rgb).astype(np.float32)
        gt_bbox_np_tir = np.array(region_tir).astype(np.float32)
        
        init_info = {'init_bbox_rgb': list(gt_bbox_np_rgb),'init_bbox_tir': list(gt_bbox_np_tir)}  # input must be (x,y,w,h)
        self.tracker.initialize_notalign(image_rgb, image_tir, init_info)
    def track(self, img_RGB, region_gt):
        '''TRACK'''
        gt_bbox_np = np.array(region_gt).astype(np.float32)
        current_info = {'gt_bbox': list(gt_bbox_np)}  
        outputs = self.tracker.track(img_RGB,current_info)
        pred_bbox = outputs['target_bbox']
        pred_score = outputs['best_score']
        return pred_bbox, pred_score
    def track_notalign(self, img, region_gt, region_gt_tir):
        '''TRACK_NOTALIGN'''
        gt_bbox_np_rgb = np.array(region_gt).astype(np.float32)
        gt_bbox_np_tir = np.array(region_gt_tir).astype(np.float32)
        current_info = {'gt_bbox_rgb': list(gt_bbox_np_rgb),'gt_bbox_tir:': list(gt_bbox_np_tir)}  
        outputs = self.tracker.track_notalign(img, self.seq_name, current_info)
        pred_bbox = outputs['target_bbox']
        pred_score = outputs['best_score']
        return pred_bbox, pred_score


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run tracker on RGBT dataset.')
    parser.add_argument('--script_name', type=str, default='bat', help='Name of tracking method(ostrack, adapter, ftuning).')
    parser.add_argument('--yaml_name', type=str, default='rgbt', help='Name of tracking method.')  
    parser.add_argument('--dataset_name', type=str, default='LasHeR', help='Name of dataset (GTOT,RGBT234,LasHeR,VTUAVST,VTUAVLT).')
    parser.add_argument('--threads', default=1, type=int, help='Number of threads')   
    parser.add_argument('--num_gpus', default=1, type=int, help='Number of gpus')
    parser.add_argument('--epoch', default=60, type=int, help='epochs of ckpt')
    parser.add_argument('--mode', default='parallel', type=str, help='sequential or parallel')
    parser.add_argument('--debug', default=0, type=int, help='to vis tracking results')
    parser.add_argument('--video', default='', type=str, help='specific video name')
    parser.add_argument('--num_cpus', default=4, type=int, help='num of cpus you want to use')
    parser.add_argument('--test_mode', default='all', type=str, help='save path')
    args = parser.parse_args()

    yaml_name = args.yaml_name
    dataset_name = args.dataset_name
    
    #cpu initial
    os.environ['OMP_NUM_THREADS'] = str(args.num_cpus)
    os.environ['OPENBLAS_NUM_THREADS'] = str(args.num_cpus)
    os.environ['MKL_NUM_THREADS'] = str(args.num_cpus)
    os.environ['VECLIB_MAXIMUM_THREADS'] = str(args.num_cpus)
    os.environ['NUMEXPR_NUM_THREADS'] = str(args.num_cpus)
    torch.set_num_threads(args.num_cpus)
    
    
    # path initialization
    seq_list = None
    if dataset_name == 'GTOT':
        seq_home = '/home/lz/Videos/GTOT'
        seq_list = [f for f in os.listdir(seq_home) if isdir(join(seq_home,f))]
        seq_list.sort()
    elif dataset_name == 'RGBT234':
        seq_home = '/root/RGB_T234'
        seq_list = [f for f in os.listdir(seq_home) if isdir(join(seq_home,f))]
        seq_list.sort()
    elif dataset_name == 'DroneT':
        seq_home = '/root/nas-resource-linkdata/DroneT'
        seq_list = [f for f in os.listdir(seq_home) if isdir(join(seq_home, f))]
        seq_list.sort()
    elif dataset_name == 'LasHeR':
        seq_home = '/root/LasHeR/TestingSet'
        with open('/root/autodl-tmp/data/testing_list.txt', 'r') as f:
            seq_list = f.read().splitlines()
        seq_list.sort()
    elif dataset_name == 'VTUAVST':
        seq_home = '/root/nas-resource-linkdata/VTUAV/test/short-term'
        with open(join(join(seq_home, 'VTUAV-ST.txt')), 'r') as f:
            seq_list = f.read().splitlines()
    elif dataset_name == 'VTUAVLT':
        seq_home = '/root/nas-resource-linkdata/VTUAV/test/long-term'
        with open(join(seq_home, 'VTUAV-LT.txt'), 'r') as f:
            seq_list = f.read().splitlines()
    elif dataset_name == 'LUART':
        seq_home = '/DATA/wangyuhang/RGBT-Tracking/SFCATrack-main/data/LUART'
        with open('/DATA/wangyuhang/RGBT-Tracking/SFCATrack-main/lib/train/data_specs/luart_testing_list.txt', 'r') as f:
            seq_list = f.read().splitlines()
        seq_list.sort()
    else:
        raise ValueError("Error dataset!")

    start = time.time()
    if args.mode == 'parallel':
        sequence_list = [(s, seq_home, dataset_name, args.yaml_name, args.num_gpus, args.epoch, args.debug, args.script_name, args.test_mode) for s in seq_list]
        multiprocessing.set_start_method('spawn', force=True)
        with multiprocessing.Pool(processes=args.threads) as pool:
            pool.starmap(run_sequence, sequence_list)
    else:
        seq_list = [args.video] if args.video != '' else seq_list
        sequence_list = [(s, seq_home, dataset_name, args.yaml_name, args.num_gpus, args.epoch, args.debug, args.script_name, args.test_mode) for s in seq_list]
        for seqlist in sequence_list:
            run_sequence(*seqlist)
    print(f"Totally cost {time.time()-start} seconds!")
