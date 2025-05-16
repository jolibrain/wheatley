from typing import List

import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer, required


class VSGD(Optimizer):
    def __init__(
        self,
        params: required,
        ghattg: float = 30.0,
        ps: float = 1e-8,
        tau1: float = 0.81,
        tau2: float = 0.9,
        lr: float = 0.1,
        weight_decay: float = 0.0,
        eps: float = 1e-8,
    ):
        """
        Args:
            ghattg: prior variance ratio between ghat and g,
                Var(ghat_t-g_t)/Var(g_t-g_{t-1}).
            ps: piror strength.
            tau1: remember rate for the gamma parameters of g
            tau2: remember rate for the gamma parameter of ghat
            lr: learning rate.
            weight_decay (float): weight decay coefficient (default: 0.0)
        """

        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        defaults = dict(
            ghattg=ghattg,
            ps=ps,
            tau1=tau1,
            tau2=tau2,
            lr=lr,
            weight_decay=weight_decay,
            eps=eps,
        )
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super(VSGD, self).__setstate__(state)

    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        # self._cuda_graph_capture_health_check()

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            mug_list = []
            step_list = []
            pa2_list = []
            pbg2_list = []
            pbhg2_list = []
            bg_list = []
            bhg_list = []

            self._init_group(
                group,
                params_with_grad,
                grads,
                mug_list,
                step_list,
                pa2_list,
                pbg2_list,
                pbhg2_list,
                bg_list,
                bhg_list,
                group["ghattg"],
                group["ps"],
            )

            vsgd(
                params_with_grad,
                grads,
                mug_list,
                step_list,
                pa2_list,
                pbg2_list,
                pbhg2_list,
                bg_list,
                bhg_list,
                group["tau1"],
                group["tau2"],
                group["lr"],
                group["weight_decay"],
                group["eps"],
            )

        return loss

    def _init_group(
        self,
        group,
        params_with_grad: List[Tensor],
        grads: List[Tensor],
        mug_list: List,
        step_list: List,
        pa2_list: List,
        pbg2_list: List,
        pbhg2_list: List,
        bg_list: List,
        bhg_list: List,
        ghattg: float,
        ps: float,
    ):
        for p in group["params"]:
            if p.grad is None:
                continue
            params_with_grad.append(p)

            grads.append(p.grad)
            state = self.state[p]

            # State initialization
            if len(state) == 0:
                for k in ["mug", "bg", "bhg"]:
                    # set a non zero small number to represent prior ignornance
                    state[k] = torch.zeros_like(p, memory_format=torch.preserve_format)
                # initialize 2*a_0 and 2*b_0 as constants
                state["pa2"] = torch.tensor(2.0 * ps + 1.0 + 1e-4)
                state["pbg2"] = torch.tensor(2.0 * ps)
                state["pbhg2"] = torch.tensor(2.0 * ghattg * ps)
                state["step"] = torch.tensor(0.0)

            mug_list.append(state["mug"])
            bg_list.append(state["bg"])
            bhg_list.append(state["bhg"])
            step_list.append(state["step"])
            pa2_list.append(state["pa2"])
            pbg2_list.append(state["pbg2"])
            pbhg2_list.append(state["pbhg2"])

    def get_current_beta1_estimate(self) -> Tensor:
        betas = []
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                bg = state["bg"]
                bhg = state["bhg"]
                betas.append((bhg / (bg + bhg)).data)
        return betas


def vsgd(
    params_with_grad: List[Tensor],
    grads: List[Tensor],
    mug_list: List[Tensor],
    step_list: List[Tensor],
    pa2_list: List[Tensor],
    pbg2_list: List[Tensor],
    pbhg2_list: List[Tensor],
    bg_list: List[Tensor],
    bhg_list: List[Tensor],
    tau1: float,
    tau2: float,
    lr: float,
    weight_decay: float,
    eps: float,
):
    for i, param in enumerate(params_with_grad):
        ghat = grads[i]
        mug = mug_list[i]
        mug1 = torch.clone(mug)
        step = step_list[i]
        step += 1
        pa2 = pa2_list[i]
        pbg2 = pbg2_list[i]
        pbhg2 = pbhg2_list[i]
        bg = bg_list[i]
        bhg = bhg_list[i]
        # weight decay following AdamW
        param.data.mul_(1 - lr * weight_decay)

        # variances of g and ghat
        if step == 1.0:
            sg = pbg2 / (pa2 - 1.0)
            shg = pbhg2 / (pa2 - 1.0)
        else:
            sg = bg / pa2
            shg = bhg / pa2
        # update muh, mug, Sigg and Sigh
        mug.copy_((ghat * sg + mug1 * shg) / (sg + shg))
        sigg = sg * shg / (sg + shg)

        # update 2*b
        mug_sq = sigg + mug**2
        bg2 = pbg2 + mug_sq - 2.0 * mug * mug1 + mug1**2
        bhg2 = pbhg2 + mug_sq - 2.0 * ghat * mug + ghat**2

        rho1 = step ** (-tau1)
        rho2 = step ** (-tau2)
        bg.mul_(1.0 - rho1).add_(bg2, alpha=rho1)
        bhg.mul_(1.0 - rho2).add_(bhg2, alpha=rho2)

        # update param
        param.data.add_(lr / (torch.sqrt(mug_sq) + eps) * mug, alpha=-1.0)
