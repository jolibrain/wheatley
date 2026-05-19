import math
import torch
from typing import List, Optional
from torch import Tensor
import torch.optim

from torch.optim.optimizer import Optimizer

version_higher = torch.__version__ >= "1.5.0"


class Anon(Optimizer):
    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-16,
        gamma=0.0,
        weight_decay=0,
        weight_decouple=False,
        fixed_decay=False,
        rectify=False,
        degenerated_to_sgd=False,
    ):

        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))

        self.degenerated_to_sgd = degenerated_to_sgd
        if (
            isinstance(params, (list, tuple))
            and len(params) > 0
            and isinstance(params[0], dict)
        ):
            for param in params:
                if "betas" in param and (
                    param["betas"][0] != betas[0] or param["betas"][1] != betas[1]
                ):
                    param["buffer"] = [[None, None, None] for _ in range(10)]

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            gamma=gamma,
            weight_decay=weight_decay,
            buffer=[[None, None, None] for _ in range(10)],
        )
        super(Anon, self).__init__(params, defaults)

        self.degenerated_to_sgd = degenerated_to_sgd
        self.weight_decouple = weight_decouple
        self.rectify = rectify
        self.fixed_decay = fixed_decay
        if self.weight_decouple:
            print("Weight decoupling enabled in Anon")
            if self.fixed_decay:
                print("Weight decay fixed")
        if self.rectify:
            print("Rectification enabled in Anon")
        print(
            "lr={}, gamma={}, eps={}, betas={}, wd={}".format(
                lr, gamma, eps, betas, weight_decay
            )
        )

    def __setstate__(self, state):
        super(Anon, self).__setstate__(state)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Anon does not support sparse gradients")

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    # Fixed learning rate
                    state["fixed_lr"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    state["t"] = 0

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                fixed_lr = state["fixed_lr"]
                beta1, beta2 = group["betas"]
                state["step"] += 1

                # perform weight decay, check if decoupled weight decay
                if self.weight_decouple:
                    if not self.fixed_decay:
                        p.data.mul_(1.0 - group["lr"] * group["weight_decay"])
                    else:
                        p.data.mul_(1.0 - group["weight_decay"])
                else:
                    if group["weight_decay"] != 0:
                        grad.add_(p.data, alpha=group["weight_decay"])

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                bias_correction1 = 1 - beta1 ** state["step"]

                t = state["t"]

                if state["step"] == 2**t:
                    if t == 0:
                        state["t"] += 1
                        bias_correction2 = 1 - beta2
                        fixed_lr.add_(
                            exp_avg_sq.div_(bias_correction2)
                            .add_(group["eps"])
                            .pow_(group["gamma"])
                            .rsqrt_()
                        )
                        exp_avg_sq.zero_()

                    else:
                        state["t"] += 1
                        bias_correction2 = 1 - beta2 ** (state["step"] / 2)
                        fixed_lr.pow_(2).reciprocal_().add_(
                            exp_avg_sq.div_(bias_correction2)
                            .add_(group["eps"])
                            .pow_(group["gamma"])
                        ).div_(2).rsqrt_()
                        exp_avg_sq.zero_()

                step_size = group["lr"] / bias_correction1
                p.data.addcmul_(exp_avg, fixed_lr, value=-step_size)

        return loss
