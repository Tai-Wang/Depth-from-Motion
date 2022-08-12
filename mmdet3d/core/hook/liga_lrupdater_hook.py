# Copyright (c) OpenMMLab. All rights reserved.
import math

import mmcv
from mmcv.runner.hooks import HOOKS
from mmcv.runner.hooks.lr_updater import StepLrUpdaterHook


@HOOKS.register_module()
class LIGALrUpdaterHook(StepLrUpdaterHook):
    """LIGA-Stereo learning rate scheme.

    This hook supports 'cosine' warmup lr updater.
    """

    def __init__(self,
                 step,
                 gamma=0.1,
                 min_lr=None,
                 by_epoch=True,
                 warmup=None,
                 warmup_iters=0,
                 warmup_ratio=0.1,
                 warmup_by_epoch=False):
        if isinstance(step, list):
            assert mmcv.is_list_of(step, int)
            assert all([s > 0 for s in step])
        elif isinstance(step, int):
            assert step > 0
        else:
            raise TypeError('"step" must be a list or integer')
        self.step = step
        self.gamma = gamma
        self.min_lr = min_lr
        # validate the "warmup" argument
        if warmup is not None:
            if warmup not in ['cosine', 'constant', 'linear', 'exp']:
                raise ValueError(
                    f'"{warmup}" is not a supported type for warming up, valid'
                    ' types are "cosine", "constant", "linear" and "exp"')
        if warmup is not None:
            assert warmup_iters > 0, \
                '"warmup_iters" must be a positive integer'
            assert 0 < warmup_ratio <= 1.0, \
                '"warmup_ratio" must be in range (0,1]'

        self.by_epoch = by_epoch
        self.warmup = warmup
        self.warmup_iters = warmup_iters
        self.warmup_ratio = warmup_ratio
        self.warmup_by_epoch = warmup_by_epoch

        if self.warmup_by_epoch:
            self.warmup_epochs = self.warmup_iters
            self.warmup_iters = None
        else:
            self.warmup_epochs = None

        self.base_lr = []  # initial lr for all param groups
        self.regular_lr = []  # expected lr if no warming up is performed

    def get_warmup_lr(self, cur_iters: int):

        def _get_warmup_lr(cur_iters, regular_lr):
            if self.warmup == 'cosine':
                k = self.warmup_ratio + (1 - self.warmup_ratio) * (
                    1 - math.cos(math.pi * cur_iters / self.warmup_iters)) / 2
                warmup_lr = [_lr * k for _lr in regular_lr]
            elif self.warmup == 'constant':
                warmup_lr = [_lr * self.warmup_ratio for _lr in regular_lr]
            elif self.warmup == 'linear':
                k = (1 - cur_iters / self.warmup_iters) * (1 -
                                                           self.warmup_ratio)
                warmup_lr = [_lr * (1 - k) for _lr in regular_lr]
            elif self.warmup == 'exp':
                k = self.warmup_ratio**(1 - cur_iters / self.warmup_iters)
                warmup_lr = [_lr * k for _lr in regular_lr]
            return warmup_lr

        if isinstance(self.regular_lr, dict):
            lr_groups = {}
            for key, regular_lr in self.regular_lr.items():
                lr_groups[key] = _get_warmup_lr(cur_iters, regular_lr)
            return lr_groups
        else:
            return _get_warmup_lr(cur_iters, self.regular_lr)
