import os
import cv2
import sys
from os.path import join, isdir, abspath, dirname
import numpy as np
import argparse
import csv
import json
prj = join(dirname(__file__), '..')
if prj not in sys.path:
    sys.path.append(prj)

import multiprocessing
import torch
from lib.train.dataset.depth_utils import get_x_frame
import time
import threading
import queue
from tqdm import tqdm
torch.set_num_threads(1)  # 限制PyTorch内部线程数

def genConfig(seq_path, set_type):
    if set_type == 'RGBT234':
        ############################################  have to refine #############################################
        RGB_img_list = sorted([seq_path + '' + p for p in os.listdir(seq_path + '') if os.path.splitext(p)[1] == '.jpg'])
        T_img_list = sorted([seq_path + '' + p for p in os.listdir(seq_path + '') if os.path.splitext(p)[1] == '.jpg'])

        RGB_gt = np.loadtxt(seq_path + '', delimiter=',')
        T_gt = np.loadtxt(seq_path + '', delimiter=',')

    elif set_type == 'DroneT':
            ############################################  have to refine #############################################
            RGB_img_list = sorted([seq_path + '' + p for p in os.listdir(seq_path + '') if
                                   os.path.splitext(p)[1] == '.jpg'])
            T_img_list = sorted([seq_path + '' + p for p in os.listdir(seq_path + '') if
                                 os.path.splitext(p)[1] == '.jpg'])

            RGB_gt = np.loadtxt(seq_path + '', delimiter=',')
            T_gt = np.loadtxt(seq_path + '', delimiter=',')

    elif set_type == 'GTOT':
        ############################################  have to refine #############################################
        RGB_img_list = sorted([seq_path + '' + p for p in os.listdir(seq_path + '') if os.path.splitext(p)[1] == '.png'])
        T_img_list = sorted([seq_path + '' + p for p in os.listdir(seq_path + '') if os.path.splitext(p)[1] == '.png'])

        RGB_gt = np.loadtxt(seq_path + '', delimiter=' ')
        T_gt = np.loadtxt(seq_path + '', delimiter=' ')

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
    
    elif set_type == 'LasHeR' or set_type == 'LasHeR-Unaligned':
        RGB_img_list = sorted([os.path.join(seq_path, 'visible', p) for p in os.listdir(os.path.join(seq_path, 'visible')) if p.endswith(".jpg")])
        T_img_list = sorted([os.path.join(seq_path, 'infrared', p) for p in os.listdir(os.path.join(seq_path, 'infrared')) if p.endswith(".jpg")])

        RGB_gt = np.loadtxt(os.path.join(seq_path, 'visible.txt'), delimiter=',')
        T_gt = np.loadtxt(os.path.join(seq_path, 'infrared.txt'), delimiter=',')

    elif 'VTUAV' in set_type:
        RGB_img_list = sorted([seq_path + '' + p for p in os.listdir(seq_path + '') if p.endswith(".jpg")])
        T_img_list = sorted([seq_path + '' + p for p in os.listdir(seq_path + '') if p.endswith(".jpg")])

        RGB_gt = np.loadtxt(seq_path + '', delimiter=' ')
        T_gt = np.loadtxt(seq_path + '', delimiter=' ')
    elif set_type == 'LUART':
        RGB_img_list = sorted([seq_path + '/NotAlign/visible/' + p for p in os.listdir(seq_path + '/NotAlign/visible') if p.endswith(".jpg")])
        T_img_list = sorted([seq_path + '/NotAlign/infrared/' + p for p in os.listdir(seq_path + '/NotAlign/infrared') if p.endswith(".jpg")])

        RGB_gt = np.loadtxt(seq_path + '/visible.txt', delimiter=',')
        T_gt = np.loadtxt(seq_path + '/infrared.txt', delimiter=',')
    elif set_type == 'MUART244':
        RGB_img_list = sorted([os.path.join(seq_path, 'visible', p) for p in os.listdir(os.path.join(seq_path, 'visible')) if p.endswith(".jpg")])
        T_img_list = sorted([os.path.join(seq_path, 'infrared', p) for p in os.listdir(os.path.join(seq_path, 'infrared')) if p.endswith(".jpg")])

        RGB_gt = np.loadtxt(os.path.join(seq_path, 'visible.txt'), delimiter=',')
        T_gt = np.loadtxt(os.path.join(seq_path, 'infrared.txt'), delimiter=',')
        
    return RGB_img_list, T_img_list, RGB_gt, T_gt


def get_x_frame_stepalign(color_path, depth_path, offset=None, init_bbox=None):
    rgb = cv2.imread(color_path)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

    tir = cv2.imread(depth_path, -1)

    tir_resized = cv2.resize(tir, (rgb.shape[1], rgb.shape[0]))
    if offset is None:
        offset = compute_offset(init_bbox[0], init_bbox[1], 
                                rgbw=rgb.shape[1], rgbh=rgb.shape[0],
                                tirw=tir.shape[1], tirh=tir.shape[0])
    tir_resized_aligned = cv2.warpPerspective(
        tir_resized, offset, (rgb.shape[1], rgb.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0)
    )
    
    img_rgb_size = cv2.merge((rgb, tir_resized_aligned))
    rgb_resized = cv2.resize(rgb, (tir.shape[1], tir.shape[0]))
    img_tir_size = cv2.merge((rgb_resized, tir))
    return img_rgb_size, img_tir_size, offset

def get_x_frame_stepalign2(color_path, depth_path, offset=None, init_bbox=None):
    rgb = cv2.imread(color_path)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

    tir = cv2.imread(depth_path, -1)

    tir_resized = cv2.resize(tir, (rgb.shape[1], rgb.shape[0]))
    if offset is None:
        offset = compute_offset(init_bbox[0], init_bbox[1], 
                                rgbw=rgb.shape[1], rgbh=rgb.shape[0],
                                tirw=tir.shape[1], tirh=tir.shape[0])
    tir_resized_aligned = cv2.warpPerspective(
        tir_resized, offset, (rgb.shape[1], rgb.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0)
    )
    
    img_rgb_size = cv2.merge((rgb, tir_resized_aligned))
    img_rgb_size2 = cv2.merge((rgb, tir, tir_resized))
    return img_rgb_size, img_rgb_size2

def compute_offset(rgb_bbox, tir_bbox, rgbw=1920, rgbh=1080, tirw=640, tirh=512):
    # 输入为numpy数组 [x1,y1,w,h] 维度均为4，返回值也为numpy数组
    scale_x = rgbw/tirw # TIR bbox放缩系数
    scale_y = rgbh/tirh
    tir_bbox_scaled = np.array([
        tir_bbox[0]*scale_x, tir_bbox[1]*scale_y,
        tir_bbox[2]*scale_x, tir_bbox[3]*scale_y
    ], dtype=np.float32)
    # 定义四个角点 (x, y)
    def bbox_to_corners(bbox):
        x, y, w, h = bbox
        return np.array([
            [x, y],           # 左上
            [x + w, y],       # 右上
            [x + w, y + h],   # 右下
            [x, y + h]        # 左下
        ], dtype=np.float32)
    pts_rgb = bbox_to_corners(rgb_bbox)
    pts_tir = bbox_to_corners(tir_bbox_scaled)
    H, _ = cv2.findHomography(pts_tir, pts_rgb, method=cv2.RANSAC, ransacReprojThreshold=3.0)
      
    return  H

def mergeHomography(H0, H_delta):
    """
    合并两个单应性矩阵，得到新的单应性矩阵 H1。

    参数:
        H0: np.ndarray, shape (3, 3)
            上一帧或初始的单应性矩阵
        H_delta: np.ndarray, shape (3, 3)
            在 H0 基础上预测的增量单应性矩阵

    返回:
        H1: np.ndarray, shape (3, 3)
            合成后的单应性矩阵
    """
    assert H0.shape == (3, 3) and H_delta.shape == (3, 3), "输入必须是 3x3 矩阵"
    H1 = H_delta @ H0   # 关键：后乘，表示在 H0 对齐后的基础上再修正
    return H1



def _result_save_path(seq_name, dataset_name, yaml_name, epoch, end_fix, script_name):
    save_name = f'{yaml_name}_ep{epoch}{end_fix}'
    save_folder = os.path.join('./output/tracking_results', script_name, save_name, dataset_name)
    return os.path.join(save_folder, f'{seq_name}.txt')


_ROUTER_STAGES = ('center', 'scale', 'refinement')
_ROUTER_CARDINALITIES = (1, 2, 3)

# ---------------------------------------------------------------------------
# Test-time policy is maintained by argparse.
# ---------------------------------------------------------------------------
def _apply_test_time_args(tracker, args):
    """Apply CLI-controlled template/TOCU settings to one tracker instance."""
    tracker.tocu_update_mode = 'relation' if args.tocu_mode == 'new' else 'legacy'

    # Dynamic template FIFO.
    tracker.template_fifo_len = args.template_fifo_len
    tracker.template_update_interval = args.template_update_interval
    tracker.template_score_thr = args.template_score_thr

    # Relation-based online homography update.
    tracker.tocu_track_score_thr = args.tocu_track_score_thr
    tracker.tocu_gain_thr = args.tocu_gain_thr
    tracker.tocu_relation_conf_thr = args.tocu_relation_conf_thr
    tracker.tocu_relation_entropy_thr = args.tocu_relation_entropy_thr
    tracker.tocu_valid_ratio_thr = args.tocu_valid_ratio_thr
    tracker.tocu_relation_mahal_thr = args.tocu_relation_mahal_thr
    tracker.tocu_commit_streak = args.tocu_commit_streak

    # Legacy reverse-tracking TOCU.
    tracker.alpha = args.tocu_legacy_alpha
    tracker.beta = args.tocu_legacy_beta


def _router_stats_dir(dataset_name, yaml_name, epoch, end_fix, script_name):
    save_name = f'{yaml_name}_ep{epoch}{end_fix}'
    return os.path.join(
        './output/tracking_results', script_name, save_name, dataset_name,
        'router_selection_stats'
    )


def _router_stats_json_path(seq_name, dataset_name, yaml_name, epoch, end_fix, script_name):
    return os.path.join(
        _router_stats_dir(dataset_name, yaml_name, epoch, end_fix, script_name),
        f'{seq_name}.json'
    )


def _empty_router_histogram():
    return {stage: {count: 0 for count in _ROUTER_CARDINALITIES} for stage in _ROUTER_STAGES}


def _update_router_histogram(histogram, frame_selection):
    if not isinstance(frame_selection, dict):
        return
    for stage in _ROUTER_STAGES:
        count = frame_selection.get(stage, None)
        if count is None:
            continue
        count = int(count)
        if count in histogram[stage]:
            histogram[stage][count] += 1


def _summarize_cardinality_counts(counts):
    total = int(sum(int(v) for v in counts.values()))
    summary = {'decisions': total}
    for count in _ROUTER_CARDINALITIES:
        value = int(counts.get(count, 0))
        summary[f'{count}_expert_count'] = value
        summary[f'{count}_expert_pct'] = 100.0 * value / total if total > 0 else 0.0
    return summary


def _build_router_sequence_summary(seq_name, histogram, total_frames):
    stage_summary = {
        stage: _summarize_cardinality_counts(histogram[stage])
        for stage in _ROUTER_STAGES
    }
    overall_counts = {count: 0 for count in _ROUTER_CARDINALITIES}
    for stage in _ROUTER_STAGES:
        for count in _ROUTER_CARDINALITIES:
            overall_counts[count] += int(histogram[stage][count])
    return {
        'sequence': seq_name,
        'total_video_frames': int(total_frames),
        'tracked_frames_excluding_initialization': max(int(total_frames) - 1, 0),
        'stages': stage_summary,
        'overall': _summarize_cardinality_counts(overall_counts),
    }


def _write_router_sequence_stats(summary, dataset_name, yaml_name, epoch, end_fix, script_name):
    stats_dir = _router_stats_dir(dataset_name, yaml_name, epoch, end_fix, script_name)
    os.makedirs(stats_dir, exist_ok=True)
    seq_name = summary['sequence']
    json_path = os.path.join(stats_dir, f'{seq_name}.json')
    csv_path = os.path.join(stats_dir, f'{seq_name}.csv')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    fieldnames = [
        'sequence', 'scope', 'decisions',
        '1_expert_count', '1_expert_pct',
        '2_expert_count', '2_expert_pct',
        '3_expert_count', '3_expert_pct',
    ]
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for scope in (*_ROUTER_STAGES, 'overall'):
            item = summary['overall'] if scope == 'overall' else summary['stages'][scope]
            writer.writerow({'sequence': seq_name, 'scope': scope, **item})
    return json_path


def _write_router_dataset_summary(dataset_name, yaml_name, epoch, end_fix, script_name):
    stats_dir = _router_stats_dir(dataset_name, yaml_name, epoch, end_fix, script_name)
    if not os.path.isdir(stats_dir):
        return None
    records = []
    for name in sorted(os.listdir(stats_dir)):
        if not name.endswith('.json') or name == 'dataset_summary.json':
            continue
        try:
            with open(os.path.join(stats_dir, name), 'r', encoding='utf-8') as f:
                records.append(json.load(f))
        except (OSError, ValueError, KeyError):
            continue
    if not records:
        return None

    columns = ['sequence', 'total_video_frames']
    for scope in (*_ROUTER_STAGES, 'overall'):
        columns.extend([f'{scope}_{count}_expert_pct' for count in _ROUTER_CARDINALITIES])
    summary_csv = os.path.join(stats_dir, 'dataset_summary.csv')
    with open(summary_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for record in records:
            row = {
                'sequence': record['sequence'],
                'total_video_frames': record.get('total_video_frames', 0),
            }
            for scope in (*_ROUTER_STAGES, 'overall'):
                item = record['overall'] if scope == 'overall' else record['stages'][scope]
                for count in _ROUTER_CARDINALITIES:
                    row[f'{scope}_{count}_expert_pct'] = item[f'{count}_expert_pct']
            writer.writerow(row)

    aggregate = {stage: {count: 0 for count in _ROUTER_CARDINALITIES} for stage in _ROUTER_STAGES}
    for record in records:
        for stage in _ROUTER_STAGES:
            for count in _ROUTER_CARDINALITIES:
                aggregate[stage][count] += int(record['stages'][stage][f'{count}_expert_count'])
    dataset_summary = _build_router_sequence_summary(
        seq_name='__dataset_total__',
        histogram=aggregate,
        total_frames=sum(int(r.get('total_video_frames', 0)) for r in records),
    )
    dataset_summary['tracked_frames_excluding_initialization'] = sum(
        int(r.get('tracked_frames_excluding_initialization', 0)) for r in records
    )
    dataset_summary['num_sequences'] = len(records)
    with open(os.path.join(stats_dir, 'dataset_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(dataset_summary, f, ensure_ascii=False, indent=2)
    return summary_csv


def _count_sequence_frames(seq_home, seq_name, dataset_name):
    """Count frames without constructing the tracker. Used only for ETA weighting."""
    seq_path = os.path.join(seq_home, seq_name)
    try:
        if dataset_name in ('LasHeR', 'LasHeR-Unaligned', 'MUART244'):
            folder = os.path.join(seq_path, 'visible')
            return sum(1 for p in os.listdir(folder) if p.endswith('.jpg'))
        if dataset_name == 'LUART':
            folder = os.path.join(seq_path, 'NotAlign', 'visible')
            return sum(1 for p in os.listdir(folder) if p.endswith('.jpg'))
        if dataset_name == 'GTOT':
            return sum(1 for p in os.listdir(seq_path) if p.endswith('.png'))
        # RGBT234 / DroneT / VTUAV and compatible layouts.
        direct_jpg = sum(1 for p in os.listdir(seq_path) if p.endswith('.jpg'))
        if direct_jpg > 0:
            return direct_jpg
    except Exception:
        pass

    # Fallback to the dataset loader if a custom layout is used.
    try:
        rgb_list, _, _, _ = genConfig(seq_path, dataset_name)
        return len(rgb_list)
    except Exception:
        # Keep the progress denominator valid even for an unknown layout.
        return 1


def _progress_monitor(progress_queue, total_frames, total_sequences, skipped_sequences=0):
    """Single parent-side progress bar; workers only report batched frame counts."""
    desc = f'Testing {total_sequences} seqs'
    if skipped_sequences:
        desc += f' ({skipped_sequences} skipped)'
    bar_format = '{l_bar}{bar}| {n_fmt}/{total_fmt} frames [{elapsed}<{remaining}, {rate_fmt}]'
    with tqdm(
        total=max(int(total_frames), 1),
        desc=desc,
        unit='frame',
        dynamic_ncols=True,
        mininterval=0.5,
        smoothing=0.1,
        bar_format=bar_format,
    ) as pbar:
        while True:
            item = progress_queue.get()
            if item is None:
                break
            if isinstance(item, tuple):
                kind = item[0]
                if kind == 'frames':
                    pbar.update(int(item[1]))
                elif kind == 'seq':
                    # Do not print from worker processes; show latest completion here.
                    _, seq_name, fps = item
                    if fps is not None:
                        pbar.set_postfix_str(f'{seq_name}, {fps:.1f} FPS', refresh=False)
                    else:
                        pbar.set_postfix_str(str(seq_name), refresh=False)
            else:
                pbar.update(int(item))
        # Avoid a visually incomplete bar from tiny counting mismatches.
        if pbar.n < pbar.total:
            pbar.update(pbar.total - pbar.n)

def run_sequence(seq_name, seq_home, dataset_name, yaml_name, save_dir, num_gpu=1, epoch=300, debug=0, script_name='adapter', test_mode='all', end_fix='', sid=0, all_seq=0, test_args=None, progress_queue=None, expected_frames=0, progress_batch=10, quiet=False):
    seq_txt = seq_name
    save_path = _result_save_path(seq_name, dataset_name, yaml_name, epoch, end_fix, script_name)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if os.path.exists(save_path):
        # This normally gets filtered in the parent. Keep it race-safe.
        if progress_queue is not None and expected_frames > 0:
            progress_queue.put(('frames', int(expected_frames)))
            progress_queue.put(('seq', seq_name, None))
        elif not quiet:
            print(f'-1 {seq_name}')
        return {'seq_name': seq_name, 'frames': 0, 'elapsed': 0.0, 'skipped': True}
    try:
        worker_name = multiprocessing.current_process().name
        worker_id = int(worker_name[worker_name.find('-') + 1:]) - 1
        gpu_id = worker_id % num_gpu
        torch.cuda.set_device(gpu_id)
    except:
        pass
    
    from lib.test.tracker.pmrl import OSTrack as TrackerClass
    import lib.test.parameter.ostrack as parameter_module
    params = parameter_module.parameters(yaml_name, save_dir, epoch, debug, seq_name)
    mmtrack = TrackerClass(params, test_mode)  # "GTOT" # dataset_name
    # All decisive template/TOCU test settings come from argparse.
    if test_args is not None:
        _apply_test_time_args(mmtrack, test_args)
    tracker = RGBT_Track(tracker=mmtrack, seq_name=seq_name)
        
    seq_path = seq_home + '/' + seq_name
    if not quiet:
        print(f'——————————Process {dataset_name} sequence: '+seq_name +'——————————————')
    RGB_img_list, T_img_list, RGB_gt, T_gt = genConfig(seq_path, dataset_name)
    if len(RGB_img_list) == len(RGB_gt):
        result = np.zeros_like(RGB_gt)
    else:
        result = np.zeros((len(RGB_img_list), 4), dtype=RGB_gt.dtype)
    result[0] = np.copy(RGB_gt[0])
    toc = 0
    frame_idx = 0
    offset = None
    merge_flag = False
    progress_acc = 0
    router_histogram = _empty_router_histogram()
    for frame_idx, (rgb_path, T_path) in enumerate(zip(RGB_img_list, T_img_list)):
        tic = cv2.getTickCount()
        if frame_idx == 0:
            # initialization
            _, _, offset = get_x_frame_stepalign(rgb_path, T_path, init_bbox = [RGB_gt[0], T_gt[0]])
            image_rgb_size, image_tir_size = get_x_frame(rgb_path, T_path, dtype=getattr(params.cfg.DATA,'XTYPE','rgbrgb_notalign_1'))
            tracker.initialize_notalign(image_rgb_size, image_tir_size, RGB_gt[0].tolist(), T_gt[0].tolist())  # xywh
        elif frame_idx > 0:
            # track
            image_rgb_size, image_rgb_size_unalign, offset_online_pre = get_x_frame_stepalign(rgb_path, T_path, offset)
            region, confidence, offset_online, merge_flag, router_expert_counts = tracker.track_notalign(
                image_rgb_size, image_rgb_size_unalign, RGB_gt[frame_idx].tolist(),
                region_gt_tir=T_gt[frame_idx].tolist()
            )  # xywh
            _update_router_histogram(router_histogram, router_expert_counts)
            result[frame_idx] = np.array(region)
            
        if merge_flag:
            # print(f'{seq_name}-{frame_idx} Update Offset')
            offset = mergeHomography(offset ,offset_online)
            
        toc += cv2.getTickCount() - tic

        if progress_queue is not None:
            progress_acc += 1
            if progress_acc >= max(int(progress_batch), 1):
                progress_queue.put(('frames', progress_acc))
                progress_acc = 0

    if progress_queue is not None and progress_acc > 0:
        progress_queue.put(('frames', progress_acc))

    toc /= cv2.getTickFrequency()
    fps = (frame_idx / toc) if toc > 0 and frame_idx > 0 else None
    if not debug and frame_idx > 0:
        np.savetxt(save_path, result.astype(int), fmt="%d")
        if not quiet:
            print('{}/{} {} , fps:{}'.format(sid, all_seq, seq_name, fps))

    router_summary = _build_router_sequence_summary(
        seq_name=seq_name, histogram=router_histogram, total_frames=frame_idx + 1
    )
    router_stats_path = _write_router_sequence_stats(
        router_summary, dataset_name, yaml_name, epoch, end_fix, script_name
    )

    if progress_queue is not None:
        progress_queue.put(('seq', seq_name, fps))
    return {
        'seq_name': seq_name,
        'frames': frame_idx + 1,
        'elapsed': toc,
        'fps': fps,
        'skipped': False,
        'router_stats_path': router_stats_path,
        'router_summary': router_summary,
    }


class RGBT_Track(object):
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
    
    def track_notalign(self, img, img_unalign, region_gt, region_gt_tir):
        '''TRACK_NOTALIGN'''
        gt_bbox_np_rgb = np.array(region_gt).astype(np.float32)
        gt_bbox_np_tir = np.array(region_gt_tir).astype(np.float32)
        current_info = {'gt_bbox_rgb': list(gt_bbox_np_rgb),'gt_bbox_tir:': list(gt_bbox_np_tir)}  
        outputs = self.tracker.track_stepalign(img, img_unalign, self.seq_name, current_info)
        pred_bbox = outputs['target_bbox']
        pred_score = outputs['best_score']
        merge_flag = outputs['merge_flag']
        offset = np.squeeze(outputs['offset'], axis=0)
        router_expert_counts = outputs.get('router_expert_counts', {})
        
        return pred_bbox, pred_score, offset, merge_flag, router_expert_counts



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run tracker on RGBT dataset.')
    parser.add_argument('--script_name', type=str, default='bat', help='Name of tracking method(ostrack, adapter, ftuning).')
    parser.add_argument('--yaml_name', type=str, default='rgbt', help='Name of tracking method.')  
    parser.add_argument('--dataset_name', type=str, default='LasHeR', help='Name of dataset (GTOT,RGBT234,LasHeR,VTUAVST,VTUAVLT).')
    parser.add_argument('--checkpoint_path', type=str, default='./output')
    parser.add_argument('--threads', default=1, type=int, help='Number of threads')   
    parser.add_argument('--num_gpus', default=1, type=int, help='Number of gpus')
    parser.add_argument('--epoch', default=60, type=int, help='epochs of ckpt')
    parser.add_argument('--mode', default='parallel', type=str, help='sequential or parallel')
    parser.add_argument('--debug', default=0, type=int, help='to vis tracking results')
    parser.add_argument('--video', default='', type=str, help='specific video name')
    parser.add_argument('--num_cpus', default=4, type=int, help='num of cpus you want to use')
    parser.add_argument('--test_mode', default='all', type=str, help='save path')
    parser.add_argument('--vis_gpus', default="0", type=str, help='Visible GPU ids, e.g. "0,1"')
    parser.add_argument('--end_fix', default="", type=str)
    parser.add_argument(
        '--tocu_mode',
        default='new',
        choices=['new', 'tocu'],
        type=str,
        help='Homography update verifier: new=relation-based single-forward method; tocu=original reverse-tracking TOCU.'
    )

    # Dynamic-template update. These args are the test-time source of truth.
    template_group = parser.add_argument_group('dynamic template update')
    template_group.add_argument('--template_fifo_len', default=10, type=int,
                                help='Maximum number of dynamic templates kept in FIFO.')
    template_group.add_argument('--template_update_interval', default=20, type=int,
                                help='Try to insert a dynamic template every N frames.')
    template_group.add_argument('--template_score_thr', default=0.60, type=float,
                                help='Insert template only when tracking score is above this threshold.')

    # Relation-based TOCU decision thresholds.
    tocu_group = parser.add_argument_group('TOCU decision thresholds')
    tocu_group.add_argument('--tocu_track_score_thr', default=0.50, type=float)
    tocu_group.add_argument('--tocu_gain_thr', default=0.01, type=float)
    tocu_group.add_argument('--tocu_relation_conf_thr', default=0.15, type=float)
    tocu_group.add_argument('--tocu_relation_entropy_thr', default=0.92, type=float)
    tocu_group.add_argument('--tocu_valid_ratio_thr', default=0.65, type=float)
    tocu_group.add_argument('--tocu_relation_mahal_thr', default=3.00, type=float)
    tocu_group.add_argument('--tocu_commit_streak', default=2, type=int,
                            help='Number of consecutive accepted frames required before committing the homography.')

    # Legacy TOCU thresholds, active only with --tocu_mode tocu.
    tocu_group.add_argument('--tocu_legacy_alpha', default=0.50, type=float)
    tocu_group.add_argument('--tocu_legacy_beta', default=0.70, type=float)
    args = parser.parse_args()

    yaml_name = args.yaml_name
    dataset_name = args.dataset_name
    
    #cpu initial
    os.environ["CUDA_VISIBLE_DEVICES"] = args.vis_gpus
    os.environ['OMP_NUM_THREADS'] = str(args.num_cpus)
    os.environ['OPENBLAS_NUM_THREADS'] = str(args.num_cpus)
    os.environ['MKL_NUM_THREADS'] = str(args.num_cpus)
    os.environ['VECLIB_MAXIMUM_THREADS'] = str(args.num_cpus)
    os.environ['NUMEXPR_NUM_THREADS'] = str(args.num_cpus)
    torch.set_num_threads(args.num_cpus)
    
    
    # path initialization
    seq_list = None
    if dataset_name == 'GTOT':
        seq_home = ''
        seq_list = [f for f in os.listdir(seq_home) if isdir(join(seq_home,f))]
        seq_list.sort()
    elif dataset_name == 'RGBT234':
        seq_home = ''
        seq_list = [f for f in os.listdir(seq_home) if isdir(join(seq_home,f))]
        seq_list.sort()
    elif dataset_name == 'DroneT':
        seq_home = ''
        seq_list = [f for f in os.listdir(seq_home) if isdir(join(seq_home, f))]
        seq_list.sort()
    elif dataset_name == 'LasHeR':
        seq_home = ''
        with open('', 'r') as f:
            seq_list = f.read().splitlines()
        seq_list.sort()
    elif dataset_name == 'LasHeR-Unaligned':    
        seq_home = '/data1/Datasets/Tracking/LasHeR-Unaligned/sequence'
        seq_list = ['blkgirlumbrella', 'blkmoto2north', 'catbrownback2bush', 'right5thflag', 'girlinrain', 'foamatgirl`srighthand', 'whitegirlinlight', 'boywalkinginsnow2', 'midgreyboyrunningcoming', 'carlightcome2', 'redroadlatboy', 'darkcarturn', 'boy2treesfindbike', 'right2ndflagformath', 'blkboytakesumbrella', 'girlafterglassdoor', 'blkboydown', "leftgirl'swhitebag", 'whiteboyrightcoccergoal', 'waitresscoming', 'bottlebetweenboy`sfeet', 'bikeboyintodark', 'blkboylefttheNo_21', 'turning1strowleft2ndboy', 'bike', 'blktribikecome', 'midboyNo_9', 'ab_pingpongball2', 'firstexercisebook', 'bikeboywithumbrella', 'guardunderthecolumn', 'whitesuvturn', 'redetricycle', 'boyinplatform', '11leftboy', '1stcol4thboy', 'bikeinrain', 'blkboyback', 'girldownstairfromlight', 'motogoesaloongS', 'whiteskirtgirlcomingfromgoal', 'blkstandboy', 'rightboy504', 'carbehindtrees', 'carturn117', 'girl`sblkbag', '7rightorangegirl', '3rdfatboy', 'girlfromlight_quezhen', '4thboywithwhite', 'advancedredcup', 'carcomingfromlight', 'mototurneast', 'boydownplatform', 'whitebikebelow', 'lastleftgirl', 'bike2left', 'shinybikeboy2left', 'leftmirrorlikesky', 'midof3girls', 'whiteridingbike', 'bikeboy', 'pinkwithblktopcup', 'umbreboyoncall', 'leftboyoutofthetroop', 'rightcomingstrongboy', 'blkboybetweenredandwhite', 'blueboy', 'bluegirlbiketurn', 'blackboy', 'blkboy`shead', 'boyride2path', 'boyunder2baskets', 'girlrightthewautress', 'leftpingpongball', 'boyshead9684', 'drillmasterfollowingatright', 'broom', 'manfromtoilet', 'littelbabycryingforahug', 'umbrellawillopen', 'redboygoright', 'ab_bolstershaking', 'basketball849', 'small-gai', 'bawgirl', 'boyaroundtrees', 'lefthyalinepaperfrontpants', 'foldedfolderatlefthand', 'darktreesboy', 'boy`headwithouthat', 'leftmirror', 'rightblkfatboyleftwhite', 'boyfromdark', 'carcominginlight', 'whiteofboys', 'hyalinepaperfrontface', 'boyatdoorturnright', 'raincarturn', 'girllongskirt', 'lefterbike', '11runtwo', 'whitecarturnleft', 'midblkgirl', 'whitegirltakingchopsticks', '1strowleftboyturning', 'lowerfoamboard', 'boytakingbasketballfollowing', 'boylefttheNo_9boy', 'rightgirltakingcup', 'rightbottlecomes', 'leftchair', 'minibusgoes2left', 'blkhairgirltakingblkbag', 'swan_0109', 'boyscomeleft', 'boyoncall', 'girlof2leaders', '3rdgrouplastboy', 'basketballathand', 'redmidboy', 'rightdarksingleman', 'leftexcersicebookyellow', 'bike2trees', 'carcomeonlight', 'ab_girlchoosesbike', 'leftrushingboy', 'middrillmaster', 'redcarcominginlight', 'carwillturn', 'pingpingpad3', 'mangetsoff', 'rightbluewhite', 'girlunderthestreetlamp', '3pinkleft', 'leftblkTboy', 'rightbike', 'bikegoindark', 'leftuphand', '1strowrightgirl3540', 'boyleftblkrunning2crowd', 'leftboy2jointhe4', 'ab_blkskirtgirl', 'shinycarcoming2', 'rightbike-gai', 'rightblkboystand', 'biketurnright', 'leftfarboycomingpicktheball', 'boytakingplate2left', 'boyinsnowfield3', 'leftopenexersicebook', 'midrunboywithwhite', 'standblkboy', 'rightcameraman', 'motowithbluetop', '10runone', 'lefthyalinepaper2rgb', '1strowrightdrillmaster', 'whiterunningboy', 'catbrown2', 'leftunderbasket', 'motocomeonlight', 'mototaking2boys306', 'ab_rightlowerredcup_quezhen', 'umbrellawillbefold', 'bikefromlight', 'moto', 'drillmaster1117', 'mandownstair', 'redtricycle', 'darkouterwhiteboy', 'shinycarcoming', 'boyruninsnow', 'leftmirrorside', 'whitecarturn683', 'blkboystand', '2runseven', 'blkboyhead', 'boy2buildings', 'leftbottle2hang', 'ballshootatthebasket3times', 'besom3', 'rainycarcome_ab', 'bluebuscoming', '1boycoming', 'midredboy', 'AQgirlwalkinrain', 'blackboyoncall', 'womanback2car', 'boy`sheadingreycol', 'blueboy421', 'carlight2', 'boy2basketballground', 'rightcar-chongT', 'boyinlight', 'runningcameragirl', '1blackteacher', 'belowdarkgirl', 'bikeboyturntimes', 'ab_whiteboywithbluebag', 'boy2trees', 'blkcaratfrontbluebus', 'rightwaiter1_quezhen', 'whitecarcomeinrain', 'large']
        seq_list.sort()
    elif dataset_name == 'VTUAVST':
        seq_home = ''
        with open(join(join(seq_home, '')), 'r') as f:
            seq_list = f.read().splitlines()
    elif dataset_name == 'VTUAVLT':
        seq_home = ''
        with open(join(seq_home, ''), 'r') as f:
            seq_list = f.read().splitlines()
    elif dataset_name == 'LUART':
        seq_home = '/data1/Datasets/Tracking/LUART/sequence'
        with open('/data1/Code/jinjiandong/unalignedtracking-baseline/lib/train/data_specs/luart_testing_list.txt', 'r') as f:
            seq_list = f.read().splitlines()
        seq_list.sort()
    elif dataset_name == 'MUART244':
        seq_home = '/data1/Datasets/Tracking/MUART244/sequence/'
        with open("/data1/Datasets/Tracking/MUART244/muart244_testinglist.txt", 'r') as f:
            seq_list = f.read().splitlines()
        seq_list.sort()
        
    else:
        raise ValueError(f"{dataset_name} is Error dataset!")

    start = time.time()

    # Apply --video before frame counting so ETA is based on the actual workload.
    if args.mode != 'parallel' and args.video != '':
        seq_list = [args.video]

    # Existing result files are removed from the ETA denominator. This prevents
    # an already-evaluated dataset from making the remaining-time estimate collapse.
    pending = []
    skipped = 0
    for sid, s in enumerate(seq_list):
        save_path = _result_save_path(
            s, dataset_name, args.yaml_name, args.epoch, args.end_fix, args.script_name
        )
        router_stats_path = _router_stats_json_path(
            s, dataset_name, args.yaml_name, args.epoch, args.end_fix, args.script_name
        )
        if os.path.exists(save_path) and os.path.exists(router_stats_path):
            skipped += 1
            continue
        frame_count = _count_sequence_frames(seq_home, s, dataset_name)
        pending.append((sid, s, max(int(frame_count), 1)))

    if len(pending) == 0:
        print(f'All {len(seq_list)} sequences already finished. Nothing to run.')
        summary_path = _write_router_dataset_summary(
            dataset_name, args.yaml_name, args.epoch, args.end_fix, args.script_name
        )
        if summary_path is not None:
            print(f'Router selection summary saved to: {summary_path}')
        print(f"Totally cost {time.time()-start:.1f} seconds!")
        sys.exit(0)

    total_frames = sum(x[2] for x in pending)
    total_sequences = len(pending)
    progress_batch = 500

    # One parent-side progress bar receives batched frame updates from all workers.
    # Therefore parallel workers never create competing tqdm bars, and ETA is based
    # on aggregate processed frames rather than the number of completed sequences.
    if args.mode == 'parallel':
        multiprocessing.set_start_method('spawn', force=True)
        manager = multiprocessing.Manager()
        progress_queue = manager.Queue(maxsize=max(args.threads * 8, 32))
        monitor = threading.Thread(
            target=_progress_monitor,
            args=(progress_queue, total_frames, total_sequences, skipped),
            daemon=True,
        )
        monitor.start()

        sequence_list = [
            (
                s, seq_home, dataset_name, args.yaml_name, args.checkpoint_path,
                args.num_gpus, args.epoch, args.debug, args.script_name,
                args.test_mode, args.end_fix, sid, len(seq_list), args,
                progress_queue, frame_count, progress_batch, True,
            )
            for sid, s, frame_count in pending
        ]

        try:
            with multiprocessing.Pool(processes=args.threads) as pool:
                pool.starmap(run_sequence, sequence_list)
        finally:
            progress_queue.put(None)
            monitor.join()
            manager.shutdown()
    else:
        progress_queue = queue.Queue()
        monitor = threading.Thread(
            target=_progress_monitor,
            args=(progress_queue, total_frames, total_sequences, skipped),
            daemon=True,
        )
        monitor.start()

        try:
            for sid, s, frame_count in pending:
                run_sequence(
                    s, seq_home, dataset_name, args.yaml_name, args.checkpoint_path,
                    args.num_gpus, args.epoch, args.debug, args.script_name,
                    args.test_mode, args.end_fix, sid, len(seq_list), args,
                    progress_queue, frame_count, progress_batch, True,
                )
        finally:
            progress_queue.put(None)
            monitor.join()

    summary_path = _write_router_dataset_summary(
        dataset_name, args.yaml_name, args.epoch, args.end_fix, args.script_name
    )
    if summary_path is not None:
        print(f'Router selection summary saved to: {summary_path}')
    print(f"Totally cost {time.time()-start:.1f} seconds!")

