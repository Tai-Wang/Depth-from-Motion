# Copyright (c) OpenMMLab. All rights reserved.
import math

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn


class NormalizeLayer(nn.Module):

    def __init__(self, type, channel, momentum=0.99):
        super().__init__()
        self.channel = channel
        self.type = type
        self.momentum = momentum
        assert type in ['scale', 'cw_scale', 'center+scale', 'cw_center+scale']

        self.channel_wise = False
        self.do_centering = False
        self.do_scaling = False
        self.scaling_method = 'abs'

        if self.type == 'scale':
            self.register_buffer('scale', torch.ones(1, 1))
            self.do_scaling = True
        elif self.type == 'cw_scale':
            self.register_buffer('scale', torch.ones(1, channel))
            self.do_scaling = self.channel_wise = True
        elif self.type == 'center+scale':
            self.register_buffer('center', torch.ones(1, 1))
            self.register_buffer('scale', torch.ones(1, 1))
            self.do_scaling = self.do_centering = True
        elif self.type == 'cw_center+scale':
            self.register_buffer('center', torch.ones(1, channel))
            self.register_buffer('scale', torch.ones(1, channel))
            self.do_scaling = self.do_centering = self.channel_wise = True
        else:
            raise ValueError('invalid normalization type')
        assert self.do_scaling or self.do_centering, \
            'at least one of scaling or centering normalization'

    def forward(self, inputs):
        if self.do_centering:
            x1 = inputs - self.center
        else:
            x1 = inputs
        if self.do_scaling:
            x2 = x1 / self.scale
        else:
            x2 = x1

        if self.training:
            self.update(inputs)
        return x2

    @torch.no_grad()
    def update(self, x):
        assert len(x.shape) == 2
        bsize = torch.tensor(x.shape[0], dtype=torch.long, device=x.device)
        dist.all_reduce(bsize, op=dist.ReduceOp.SUM)
        if bsize <= 10:
            return

        if self.do_centering:
            sum_x = torch.sum(x, dim=0, keepdim=True)
            dist.all_reduce(sum_x, op=dist.ReduceOp.SUM)
            new_center = sum_x / torch.clamp(bsize, min=1)
            if not self.channel_wise:
                new_center = new_center.mean(dim=-1, keepdim=True)

            self.center *= self.momentum
            self.center += new_center * (1 - self.momentum)

            x = x - new_center

        if self.do_scaling:
            if self.scaling_method == 'abs':
                sum_x = torch.sum(x.abs(), dim=0, keepdim=True)
                dist.all_reduce(sum_x, op=dist.ReduceOp.SUM)
                new_scale = sum_x / torch.clamp(bsize, min=1)
            elif self.scaling_method == 'std':
                sum_x_sq = torch.sum(x**2, dim=0, keepdim=True)
                dist.all_reduce(sum_x_sq, op=dist.ReduceOp.SUM)
                new_scale = torch.sqrt(sum_x_sq / torch.clamp(bsize, min=1))
                # if using std, normalize to 0.667
                new_scale = new_scale
            else:
                raise ValueError('invalid scale method')
            if not self.channel_wise:
                new_scale = new_scale.mean(dim=-1, keepdim=True)

            self.scale *= self.momentum
            self.scale += new_scale * (1 - self.momentum)


class WeightedL2WithSigmaLoss(nn.Module):

    def __init__(self, code_weights: list = None):
        super(WeightedL2WithSigmaLoss, self).__init__()
        if code_weights is not None:
            self.code_weights = np.array(code_weights, dtype=np.float32)
            self.code_weights = torch.from_numpy(self.code_weights).cuda()
        else:
            self.code_weights = None

    @staticmethod
    def l2_loss(diff, sigma=None):
        if sigma is None:
            loss = 0.5 * diff**2
        else:
            loss = 0.5 * (diff / torch.exp(sigma))**2 + math.log(
                math.sqrt(6.28)) + sigma

        return loss

    def forward(self,
                input: torch.Tensor,
                target: torch.Tensor,
                weights: torch.Tensor = None,
                sigma: torch.Tensor = None):
        # ignore nan targets
        target = torch.where(torch.isnan(target), input, target)

        diff = input - target
        # code-wise weighting
        if self.code_weights is not None:
            diff = diff * self.code_weights  # .view(1, 1, -1)

        loss = self.l2_loss(diff, sigma=sigma)

        # anchor-wise weighting
        if weights is not None:
            assert weights.shape == loss.shape[:-1]
            weights = weights.unsqueeze(-1)
            assert len(loss.shape) == len(weights.shape)
            loss = loss * weights

        return loss
