import torch
from .util import enable_running_stats, disable_running_stats
import contextlib
from torch.distributed import ReduceOp
import math
import numpy as np
import random

class vSAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, model, rho, adaptive=False, perturb_eps=1e-12, hp_win=[50,5], max_p=0.8, hp1=[0.01,0.01], hp2=0.7, last_layer = 2, grad_reduce='mean', **kwargs):
        defaults = dict(adaptive=adaptive, **kwargs)
        super(vSAM, self).__init__(params, defaults)
        self.model = model
        self.base_optimizer = base_optimizer
        self.param_groups = self.base_optimizer.param_groups
        self.adaptive = adaptive
        self.perturb_eps = perturb_eps
        self.hp_win = hp_win
        self.hp1 = hp1
        self.hp2 = hp2
        self.hp2_ = hp2
        self.rho = rho
        self.epoch = 0
        self.flag_batch_num = 1

        self.og_norm = 0
        self.psf_norm = 0
        self.vars_ = torch.zeros(self.hp_win[1]).cuda()

        self.sampling_number = 0
        self.sampling_probability = 0.5

        self.psf_norm_all_th = torch.zeros(self.hp_win[0]).cuda()
        self.var_all_th = torch.zeros(self.hp_win[0]).cuda()
        self.psf_norm_all_th_last = torch.zeros(self.hp_win[0]).cuda()
        self.var_all_th_last = torch.zeros(self.hp_win[0]).cuda()
        self.psf_norm_all_th_o = torch.zeros(self.hp_win[0]-1).cuda()
        self.var_all_th_o = torch.zeros(self.hp_win[0]-1).cuda()

        self.base_sr = 1
        self.K = self.hp_win[0]
        self.max_sr = int(self.hp_win[0] * max_p)
        self.sr_now = 0
        self.time_dic = {}
        self.last_layer = last_layer

        for group in self.param_groups:
            self.p_len = len(group["params"])

        # set up reduction for gradient across workers
        if grad_reduce.lower() == 'mean':
            if hasattr(ReduceOp, 'AVG'):
                self.grad_reduce = ReduceOp.AVG
                self.manual_average = False
            else:  # PyTorch <= 1.11.0 does not have AVG, need to manually average across processes
                self.grad_reduce = ReduceOp.SUM
                self.manual_average = True
        elif grad_reduce.lower() == 'sum':
            self.grad_reduce = ReduceOp.SUM
            self.manual_average = False
        else:
            raise ValueError('"grad_reduce" should be one of ["mean", "sum"].')


    @torch.no_grad()
    def perturb_weights(self):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = self.rho / (grad_norm + self.perturb_eps)
            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_g"] = p.grad.clone()
                self.state[p]["old_p"] = p.data.clone()
                e_w = p.grad * scale.to(p)
                p.add_(e_w)


    @torch.no_grad()
    def unperturb(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]
                self.state[p]["sam_g"] = p.grad.clone()
                self.state[p]["so_g"] = (self.state[p]["sam_g"] - self.state[p]["old_g"])


    def get_var(self, win, w_w):
        sgd_norm_now = self.psf_norm_all_th
        win_w = int(len(sgd_norm_now) / w_w)
        sort_data = torch.sort(sgd_norm_now).values
        for j in range(w_w):
            self.vars_[j] = (torch.var(sort_data[j * win_w: (j + 1) * win_w]))
        return self.vars_.mean()

    @torch.no_grad()
    def _sync_grad(self):
        if torch.distributed.is_initialized():  # synchronize final gardients
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is None: continue
                    if self.manual_average:
                        torch.distributed.all_reduce(p.grad, op=self.grad_reduce)
                        world_size = torch.distributed.get_world_size()
                        p.grad.div_(float(world_size))
                    else:
                        torch.distributed.all_reduce(p.grad, op=self.grad_reduce)
        return

    @torch.no_grad()
    def _grad_norm(self, by=None):
        if not by:
            norm = torch.norm(
                torch.stack([
                    p.grad.norm(p=2)
                    for group in self.param_groups for p in group["params"]
                    if p.grad is not None
                ]),
                p=2
            )
        else:
            norm = torch.norm(
                torch.stack([
                    self.state[p][by].norm(p=2)
                    for group in self.param_groups for p in group["params"]
                    if p.grad is not None
                ]),
                p=2
            )
        return norm

    @torch.no_grad()
    def _grad_norm_l1(self, by=None, weight_adaptive=False):
        if not by:
            norm = torch.norm(
                torch.stack([
                    ((torch.abs(p.data) if weight_adaptive else 1.0) * p.grad).norm(p=2)
                    for group in self.param_groups for p in group["params"]
                    if p.grad is not None
                ]),
                p=1
            )
        else:
            norm = torch.norm(
                torch.stack([
                    ((torch.abs(p.data) if weight_adaptive else 1.0) * self.state[p][by]).norm(p=2)
                    for group in self.param_groups for p in group["params"]
                    if p.grad is not None
                ]),
                p=1
            )
        return norm

    def _grad_l2_norm_layer_N(self, by=None, N=10):
        if not by:
            norm = torch.norm(
                torch.stack([
                    (1.0 * p.grad).norm(p=2)
                    for group in self.param_groups for i, p in enumerate(group["params"])
                    if p.grad is not None and i > self.p_len - (N + 1)
                ]),
                p=2
            )
        else:
            norm = torch.norm(
                torch.stack([
                    (1.0 * self.state[p][by]).norm(p=2)
                    for group in self.param_groups for i, p in enumerate(group["params"])
                    if p.grad is not None and i > self.p_len - (N + 1)
                ]),
                p=2
            )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups

    def maybe_no_sync(self):
        if torch.distributed.is_initialized():
            return self.model.no_sync()
        else:
            return contextlib.ExitStack()

    @torch.no_grad()
    def set_closure(self, loss_fn, inputs, targets, **kwargs):
        # create self.forward_backward_func, which is a function such that
        # self.forward_backward_func() automatically performs forward and backward passes.
        # This function does not take any arguments, and the inputs and targets data
        # should be pre-set in the definition of partial-function

        def get_grad():
            self.base_optimizer.zero_grad()
            with torch.enable_grad():
                outputs = self.model(inputs)
                loss = loss_fn(outputs, targets, **kwargs)
            loss.backward()
            return outputs

        self.forward_backward_func = get_grad

    def opt_sam(self, get_grad):
        self.hp2_ = 1
        self.sr_now += 1
        outputs = get_grad()

        self.perturb_weights()
        disable_running_stats(self.model)

        get_grad()

        self.unperturb()


        self.og_norm = self._grad_l2_norm_layer_N(by='old_g', N=self.last_layer)
        self.psf_norm = self._grad_l2_norm_layer_N(by='so_g', N=self.last_layer)

        self.psf_norm_all_th = torch.cat((self.psf_norm_all_th, torch.unsqueeze(self.psf_norm, 0)))
        self.psf_norm_all_th = self.psf_norm_all_th[self.psf_norm_all_th.shape[0] - self.K :]

        self.psf_norm_all_th_last = torch.cat((self.psf_norm_all_th_last, torch.unsqueeze(self.psf_norm / self.og_norm, 0)))
        self.psf_norm_all_th_last = self.psf_norm_all_th_last[self.psf_norm_all_th_last.shape[0] - self.K:]

        self.sr_var = self.get_var(self.hp_win[0], self.hp_win[1])
        self.var_all_th = torch.cat((self.var_all_th, torch.unsqueeze(self.sr_var, 0)))

        self.base_optimizer.step()
        enable_running_stats(self.model)

        self.sampling_number += 1
        return outputs

    def opt_sgd(self, get_grad):
        outputs = get_grad()
        if self.hp2_ > pow(self.hp2, 6):
            self.hp2_ = self.hp2_ * self.hp2
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is None: continue
                    p.grad = p.grad + self.hp2_ * self.state[p]["so_g"]
            # 更新参数
            self.base_optimizer.step()
        else:
            self.base_optimizer.step()

        return outputs


    def step(self, closure=None):
        get_grad = self.forward_backward_func

        if self.epoch < 3:
            outputs = self.opt_sam(get_grad)
            self.sr_now = 1
            return outputs, self.sampling_number
        else:

            if self.flag_batch_num % self.K == 0:
                self.var_all_th = self.var_all_th[self.var_all_th.shape[0] - self.K:]

                for li in range(self.hp_win[0] - 1):
                    self.psf_norm_all_th_o[li] = (self.psf_norm_all_th_last[li + 1] - self.psf_norm_all_th_last[li]) / self.psf_norm_all_th_last[li]
                    self.var_all_th_o[li] = (self.var_all_th[li + 1] - self.var_all_th[li]) / self.var_all_th[li]
                sr_norm_mean = self.psf_norm_all_th_o.mean()
                if sr_norm_mean > 1: sr_norm_mean = 1
                if sr_norm_mean < -1: sr_norm_mean = -1

                sr_var_mean = self.var_all_th_o.mean()
                if sr_var_mean > 1: sr_var_mean = 1
                if sr_var_mean < -1: sr_var_mean = -1
                self.base_sr = self.base_sr * (1 + self.hp1[0] * sr_norm_mean + self.hp1[1] * sr_var_mean)

                if self.base_sr > self.max_sr: self.base_sr = self.max_sr
                if self.base_sr < 1: self.base_sr = 1

                self.sampling_probability = self.base_sr / self.K
                self.sr_now = 0

            if (self.sr_now < int(self.base_sr) and (random.random() >= 1 - self.sampling_probability)) or \
                    (self.K-self.flag_batch_num % self.K <= int(self.base_sr)-self.sr_now):
                # sam
                outputs = self.opt_sam(get_grad)
            else:
                # sgd
                outputs = self.opt_sgd(get_grad)

        self.flag_batch_num += 1
        return outputs, self.sampling_number