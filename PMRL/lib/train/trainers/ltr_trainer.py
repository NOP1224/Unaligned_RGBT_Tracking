import os
import datetime
import time
from collections import OrderedDict

import torch
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data.distributed import DistributedSampler

from lib.train.trainers import BaseTrainer
from lib.train.admin import AverageMeter, StatValue, TensorboardWriter
from lib.utils.misc import get_world_size, is_main_process, is_dist_avail_and_initialized


class LTRTrainer(BaseTrainer):
    def __init__(self, actor, loaders, optimizer, settings, lr_scheduler=None, use_amp=False):
        super().__init__(actor, loaders, optimizer, settings, lr_scheduler)
        self._set_default_settings()
        self.stats = OrderedDict({loader.name: None for loader in self.loaders})

        self.tensorboard_writer = None
        if is_main_process():
            tensorboard_writer_dir = os.path.join(
                self.settings.env.tensorboard_dir, self.settings.project_path
            )
            os.makedirs(tensorboard_writer_dir, exist_ok=True)
            self.tensorboard_writer = TensorboardWriter(
                tensorboard_writer_dir, [loader.name for loader in loaders]
            )

        self.move_data_to_gpu = getattr(settings, 'move_data_to_gpu', True)
        self.settings = settings
        self.use_amp = bool(use_amp)
        self.scaler = GradScaler() if self.use_amp else None

    def _set_default_settings(self):
        defaults = {'print_interval': 10, 'print_stats': None, 'description': ''}
        for parameter, default_value in defaults.items():
            if getattr(self.settings, parameter, None) is None:
                setattr(self.settings, parameter, default_value)

    @staticmethod
    def _scalar(value):
        if torch.is_tensor(value):
            return value.detach().float().mean()
        return torch.tensor(float(value), dtype=torch.float32)

    def _reduce_stats(self, new_stats: OrderedDict) -> OrderedDict:
        """Average one scalar vector across ranks with one collective per batch."""
        if not is_dist_avail_and_initialized():
            return OrderedDict((key, float(self._scalar(value).item())) for key, value in new_stats.items())
        keys = list(new_stats.keys())
        values = torch.stack([
            self._scalar(new_stats[key]).to(device=self.device) for key in keys
        ])
        torch.distributed.all_reduce(values, op=torch.distributed.ReduceOp.SUM)
        values /= float(get_world_size())
        return OrderedDict((key, float(values[index].item())) for index, key in enumerate(keys))

    def cycle_dataset(self, loader):
        self.actor.train(loader.training)
        torch.set_grad_enabled(loader.training)

        if self.settings.fix_bn:
            if is_main_process() and self.epoch == 1:
                print("#####----fix bn-------#####")
            self.actor.fix_bns()

        self._init_timing()
        batch_size = 0
        for i, data in enumerate(loader, 1):
            self.data_read_done_time = time.time()
            if self.move_data_to_gpu:
                data = data.to(self.device)
            self.data_to_gpu_time = time.time()

            data['epoch'] = self.epoch
            data['settings'] = self.settings
            if not self.use_amp:
                loss, stats = self.actor(data)
            else:
                with autocast():
                    loss, stats = self.actor(data)

            if loader.training:
                self.optimizer.zero_grad(set_to_none=True)
                if not self.use_amp:
                    loss.backward()
                    if self.settings.grad_clip_norm > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.actor.net.parameters(), self.settings.grad_clip_norm
                        )
                    self.optimizer.step()
                else:
                    self.scaler.scale(loss).backward()
                    if self.settings.grad_clip_norm > 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.actor.net.parameters(), self.settings.grad_clip_norm
                        )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

            batch_size = data['template_images'].shape[loader.stack_dim]
            reduced_stats = self._reduce_stats(stats)
            self._update_stats(reduced_stats, batch_size * get_world_size(), loader)
            self._print_stats(i, loader, batch_size)

        if is_main_process() and batch_size > 0:
            epoch_time = self.prev_time - self.start_time
            print("Epoch Time: " + str(datetime.timedelta(seconds=epoch_time)))
            print("Avg Data Time: %.5f" % (self.avg_date_time / self.num_frames * batch_size * get_world_size()))
            print("Avg GPU Trans Time: %.5f" % (self.avg_gpu_trans_time / self.num_frames * batch_size * get_world_size()))
            print("Avg Forward Time: %.5f" % (self.avg_forward_time / self.num_frames * batch_size * get_world_size()))

    def train_epoch(self):
        for loader in self.loaders:
            if self.epoch % loader.epoch_interval == 0:
                if isinstance(loader.sampler, DistributedSampler):
                    loader.sampler.set_epoch(self.epoch)
                self.cycle_dataset(loader)

        self._stats_new_epoch()
        if is_main_process():
            self._write_tensorboard()

    def _init_timing(self):
        self.num_frames = 0
        self.start_time = time.time()
        self.prev_time = self.start_time
        self.avg_date_time = 0.0
        self.avg_gpu_trans_time = 0.0
        self.avg_forward_time = 0.0

    def _update_stats(self, new_stats: OrderedDict, global_batch_size, loader):
        if loader.name not in self.stats or self.stats[loader.name] is None:
            self.stats[loader.name] = OrderedDict(
                (name, AverageMeter()) for name in new_stats.keys()
            )

        if loader.training:
            for i, lr in enumerate(self.lr_scheduler.get_last_lr()):
                var_name = 'LearningRate/group{}'.format(i)
                if var_name not in self.stats[loader.name]:
                    self.stats[loader.name][var_name] = StatValue()
                self.stats[loader.name][var_name].update(lr)

        for name, value in new_stats.items():
            if name not in self.stats[loader.name]:
                self.stats[loader.name][name] = AverageMeter()
            self.stats[loader.name][name].update(value, global_batch_size)

    def _print_stats(self, i, loader, local_batch_size):
        global_batch_size = local_batch_size * get_world_size()
        self.num_frames += global_batch_size
        current_time = time.time()
        elapsed = max(current_time - self.prev_time, 1e-8)
        batch_fps = global_batch_size / elapsed
        average_fps = self.num_frames / max(current_time - self.start_time, 1e-8)
        prev_frame_time = self.prev_time
        self.prev_time = current_time

        self.avg_date_time += self.data_read_done_time - prev_frame_time
        self.avg_gpu_trans_time += self.data_to_gpu_time - self.data_read_done_time
        self.avg_forward_time += current_time - self.data_to_gpu_time

        if not is_main_process():
            return
        if i % self.settings.print_interval != 0 and i != len(loader):
            return

        print_str = '[%s: %d, %d / %d] ' % (loader.name, self.epoch, i, len(loader))
        print_str += 'FPS: %.1f (%.1f)  ,  ' % (average_fps, batch_fps)
        print_str += 'DataTime: %.3f (%.3f)  ,  ' % (
            self.avg_date_time / self.num_frames * global_batch_size,
            self.avg_gpu_trans_time / self.num_frames * global_batch_size,
        )
        print_str += 'ForwardTime: %.3f  ,  ' % (
            self.avg_forward_time / self.num_frames * global_batch_size
        )
        print_str += 'TotalTime: %.3f  ,  ' % (
            (current_time - self.start_time) / self.num_frames * global_batch_size
        )
        for name, value in self.stats[loader.name].items():
            if self.settings.print_stats is None or name in self.settings.print_stats:
                if hasattr(value, 'avg'):
                    print_str += '%s: %.5f  ,  ' % (name, value.avg)

        line = print_str[:-5]
        print(line)
        with open(self.settings.log_file, 'a') as file:
            file.write(line + '\n')

    def _stats_new_epoch(self):
        for loader in self.loaders:
            if loader.training:
                try:
                    lr_list = self.lr_scheduler.get_last_lr()
                except Exception:
                    lr_list = self.lr_scheduler._get_lr(self.epoch)
                for i, lr in enumerate(lr_list):
                    var_name = 'LearningRate/group{}'.format(i)
                    if self.stats[loader.name] is None:
                        continue
                    if var_name not in self.stats[loader.name]:
                        self.stats[loader.name][var_name] = StatValue()
                    self.stats[loader.name][var_name].update(lr)

        for loader_stats in self.stats.values():
            if loader_stats is None:
                continue
            for stat_value in loader_stats.values():
                if hasattr(stat_value, 'new_epoch'):
                    stat_value.new_epoch()

    def _write_tensorboard(self):
        if self.tensorboard_writer is None:
            return
        if self.epoch == 1:
            self.tensorboard_writer.write_info(
                self.settings.script_name, self.settings.description
            )
        self.tensorboard_writer.write_epoch(self.stats, self.epoch)
