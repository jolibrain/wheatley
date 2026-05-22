# from pp.solution import Solution

import torch
import math
import numpy as np


class InvalidMoveException(Exception):
    def __init__(self, source, dest):
        self.source = source
        self.dest = dest
        self.msg = f"Invalid Move from {source} to {dest}"
        super().__init__(self.msg)

    def __str__(self):
        return self.msg


class InvalidSelectionException(Exception):
    def __init__(self, nid):
        self.nid = nid
        self.msg = f"Invalid Move selection of {nid}"
        super().__init__(self.msg)

    def __str__(self):
        return self.msg


class State:
    def __init__(
        self,
        env,
        start,
        goal_nids,
        max_n_steps,
        n_nodes,
        n_breaks=0,
        nonchrono=None,
    ):
        self.env = env
        self.start = start
        self.goal_nids = list(goal_nids)
        self.goal = self.goal_nids[0] if self.goal_nids else None
        self.max_n_steps = max_n_steps
        self.n_nodes = n_nodes
        self.nonchrono = nonchrono

        # others
        self.n_breaks_init = n_breaks  # number of remaining breaks in walls
        self.n_breaks = self.n_breaks_init
        self.goal_available = torch.ones(len(self.goal_nids), dtype=torch.bool)
        self.just_reached_goal = False
        self.step_cost_multiplier = 1.0

        self.reset()

    def reset(self):
        self.n_steps = 0
        self.n_forced = 0
        self.n_breaks = self.n_breaks_init
        self.just_reached_goal = False
        self.step_cost_multiplier = 1.0
        self.n_neigh = 0
        if self.nonchrono in ["wp", "wpr"]:
            self.path = None
            # self.mask = torch.ones(self.n_nodes, dtype=torch.bool)
            self.mask = self.env.accessible.clone()
            self.selected = torch.zeros(self.n_nodes)
            self.selected[self.start] = 1
            self.visited = torch.zeros(self.n_nodes)
            self.visited[self.start] = 1
            if self.goal is not None:
                self.selected[self.goal] = 1
                self.mask[self.goal] = False

            self.mask[self.start] = False
        elif self.nonchrono == "path":
            self.mask = torch.ones(self.n_nodes, dtype=torch.bool)
            self.mask[self.start] = False
            # below codex idea:
            # coords = self.env.graph.get_ndata("norm_coord").float()
            # s = coords[self.start]
            # g = coords[self.goal]
            # direction = g - s
            # proj = (coords - s) @ direction / (direction.dot(direction) + 1e-6)
            # self.selected = 5.0 - 4.0 * proj + 0.1 * torch.randn(self.n_nodes)
            # end of codex idaa
            self.selected = torch.tensor([-1.0] * self.n_nodes)
            self.selected[self.start] = 5.0
            if self.goal is not None:
                self.selected[self.goal] = 1.0
                self.mask[self.goal] = False
            self.selected = (self.selected - self.selected.mean()) / (
                self.selected.std() + 1e-6
            )
            self.update_partial_sol()
        else:
            self.pos = self.start
            self.path = []
            self.visited = torch.zeros(self.n_nodes)
            self.visited[self.pos] = 1
            self.cur_pos = torch.zeros(self.n_nodes)
            self.cur_pos[self.pos] = 1

    def compute_path_nonchrono2(self):
        # normalize selected here
        # self.selected = torch.nn.functional.layer_norm(self.selected, (self.n_nodes,))

        if self.selected[self.start] < self.selected[self.goal]:
            minv = self.selected[self.start]
            maxv = self.selected[self.goal]
            inverted = False
        else:
            minv = self.selected[self.goal]
            maxv = self.selected[self.start]
            inverted = True

        between_start_goal = torch.where(
            torch.logical_and(
                self.selected >= minv,
                self.selected <= maxv,
            )
        )[0]

        order_between_start_goal = between_start_goal[
            torch.sort(self.selected[between_start_goal], descending=inverted)[1]
        ]
        order_all = torch.sort(self.selected, descending=inverted)[1]

        sol_path = []
        n_neigh_all = 0
        n_neigh_between = 0
        path_graph_all = []
        path_graph_between = []
        node_in_path_between = torch.zeros_like(self.selected)
        # node_in_path_between[self.start] = 1
        # node_in_path_between[self.goal] = 1
        # for i in range(1, order_between_start_goal.shape[0]):
        #     n = order_between_start_goal[i].item()
        #     prev = order_between_start_goal[i - 1].item()
        #     neigh = self.env.pos_neigh_nodes(prev)
        #     if n in neigh:
        #         if n != self.goal:
        #             n_neigh_between += 1
        #             node_in_path_between[n] = 1
        #         if sol_path is not None:
        #             sol_path.append(n)
        #     else:
        #         sol_path = None
        #     path_graph_between.append([prev, n])

        # neighborhood from start and goal on path only
        for i in range(1, order_between_start_goal.shape[0]):
            n = order_between_start_goal[i].item()
            prev = order_between_start_goal[i - 1].item()
            neigh = self.env.pos_neigh_nodes(prev)
            if n in neigh:
                if n != self.goal:
                    n_neigh_between += 1
                    node_in_path_between[n] = 1
                if sol_path is not None:
                    sol_path.append(n)
            #    path_graph_between.append([prev, n])
            else:
                sol_path = None
                break
            path_graph_between.append([prev, n])

        for i in range(order_between_start_goal.shape[0] - 1, 0, -1):
            n = order_between_start_goal[i - 1].item()
            nextn = order_between_start_goal[i].item()
            neigh = self.env.pos_neigh_nodes(n)
            if nextn in neigh:
                if n != self.start:
                    n_neigh_between += 1
                    node_in_path_between[n] = 1
            else:
                break
            path_graph_between.append([n, nextn])

        for i in range(1, order_all.shape[0]):
            n = order_all[i].item()
            prev = order_all[i - 1].item()
            neigh = self.env.pos_neigh_nodes(prev)
            if n in neigh:
                n_neigh_all += 1
            path_graph_all.append([prev, n])

        if sol_path is not None:
            if len(sol_path) == 0:
                sol_path = None

        neighborhood = 0.0
        # for i in range(0, order_between_start_goal.shape[0]):
        #     n = order_between_start_goal[i].item()
        #     neighs = self.env.pos_neigh_nodes(n)
        #     for neigh in neighs:
        #         pos_neigh = (order_between_start_goal == neigh).nonzero(as_tuple=True)[
        #             0
        #         ]
        #         if pos_neigh.numel() != 0:
        #             dist = pos_neigh[0].item() - i
        #             if dist > 0:
        #                 neighborhood += (1.0 / dist) * (1.0 / dist)

        return (
            sol_path,
            n_neigh_all,
            n_neigh_between,
            path_graph_all,
            path_graph_between,
            node_in_path_between,
            neighborhood,
        )

    def compute_solution_path(self):
        # choose randomly among least visisted in order to avoid failure
        pos = self.start
        self.visited = torch.zeros(self.n_nodes)
        self.visited[pos] = 1
        path = []
        while True:
            neigh = self.env.pos_neigh_nodes(pos)
            suc = []
            # TODO parallelize
            for n in neigh:
                if not self.visited[n] and self.selected[n]:
                    suc.append(n)
            if len(suc) == 1:
                path.append(suc[0])
                if self.goal is not None:
                    if suc[0] == self.goal:
                        return path
                pos = suc[0]
                self.visited[suc[0]] = 1
                continue
            if len(suc) == 0:
                return None
            if len(suc) > 1:
                return None

    def get_mask(self):
        return self.mask

    def succeeded(self):
        if self.nonchrono is not None:
            return self.path is not None
        else:
            return bool(getattr(self.env, "all_goals_consumed", lambda: False)())

    def force_select(self):
        n_forced = 0
        for nt in torch.where(torch.logical_not(self.selected))[0]:
            if self.selected[self.env.pos_neigh_nodes(nt.item())].sum() == 2:
                self.selected[nt] = 1
                n_forced += 1
        return n_forced

    def compute_mask(self):
        selected = torch.where(self.selected == 1.0)[0]
        mask = self.env.accessible.clone()
        # cannot select multiple times
        mask[selected] = False
        for nt in selected:
            n = nt.item()
            neighs = self.env.pos_neigh_nodes(n)
            selected_neighs = self.selected[neighs]
            if selected_neighs.sum() == 2:
                # mask all  neighs, selected and not selected
                # could be extended to non selected nodes to help force connect
                mask[neighs] = False
            # elif selected_neighs.sum() == 1:
            #     # complicated case : should mask neighbors of both this neighbor and me
            #     selected_neigh = torch.where(selected_neighs)[0][0].item()
            #     nn = self.env.pos_neigh_nodes(selected_neigh)
            #     for i in nn:
            #         neigh_i = self.env.pos_neigh_nodes(i)
            #         if n in neigh_i:
            #             mask[i] = False
        return mask

    def failed(self):
        # print('non chrono in failed(): ', self.nonchrono)
        if self.nonchrono == "wp":
            # print('mask sum==0 in failed(): ', self.mask.sum() == 0, ' / path is None=', self.path is None)

            return self.mask.sum() == 0 and self.path is None
        elif self.nonchrono in ["wpr", "path"]:
            return self.n_steps == self.max_n_steps and self.path is None
        else:
            if self.pos == self.goal:
                return False
            return (
                self.n_steps == self.max_n_steps
                or self.env.transition_model.get_mask(self).sum() == 0
            )

    def done(self):
        return self.failed() or self.succeeded()

    def move_to(self, nid):
        self.n_steps += 1
        if self.env.graph.find_edge("wall", self.pos, nid) is not None:
            self.n_breaks -= 1
        self.cur_pos[self.pos] = 0
        self.pos = nid
        self.path.append(nid)
        self.visited[self.pos] += 1
        self.cur_pos[self.pos] = 1

    def set_goal_status(self, goal_status):
        if goal_status is None:
            self.goal_available = torch.zeros(0, dtype=torch.bool)
        else:
            self.goal_available = torch.as_tensor(goal_status, dtype=torch.bool).clone()

    def on_goal_consumed(self, goal_idx):
        if 0 <= goal_idx < len(self.goal_available):
            self.goal_available[goal_idx] = False
        self.just_reached_goal = True

    def clear_goal_contact(self):
        self.just_reached_goal = False

    def select(self, nid):
        self.n_steps += 1
        self.selected[nid] = 1
        # self.n_forced = self.force_select()
        self.mask = self.compute_mask()
        self.path = self.compute_solution_path()

    def select_wpr(self, nid):
        self.n_steps += 1
        if self.selected[nid] == 1:
            self.selected[nid] = 0
        else:
            self.selected[nid] = 1
            self.n_forced = self.force_select()
        # self.mask = self.compute_mask()
        self.path = self.compute_solution_path()

    def update_partial_sol(self):
        (
            sol_path,
            n_neigh_all,
            n_neigh_between,
            path_graph_all,
            path_graph_between,
            node_in_path_between,
            neighborhood,
        ) = self.compute_path_nonchrono2()

        NB = True  # definitely True seems better
        PB = True
        if NB:
            self.n_neigh = n_neigh_between
        else:
            self.n_neigh = n_neigh_all

        # NEW NEWBORHOOD
        # self.n_neigh = neighborhood
        if PB:
            self.path_graph = path_graph_between
        else:
            self.path_graph = path_graph_all
        self.node_in_path = node_in_path_between
        self.path = sol_path

    def update_path(self, nvals):
        self.n_steps += 1
        # ADDITIVE
        # self.selected += nvals
        # SIMPLE => do not work

        self.selected += nvals[: self.n_nodes]
        # self.selected = (self.selected - self.selected.mean()) / (
        #     self.selected.std() + 1e-6
        # )
        # self.selected += nvals / (nvals.std() + 1e-6)
        # self.selected[self.start] = 5.0
        # self.selected[self.goal] = 1.0
        # max_v = self.selected.max()
        # min_v = self.selected.min()
        # self.selected -= min_v
        # self.selected /= max_v - min_v
        # self.selected *= 6
        # self.selected -= 1
        # self.selected = ((self.selected - min_v) / (max_v - min_v)) * 8 - 1
        # self.selected[self.start] = 5.0
        # self.selected[self.goal] = 1.0

        # self.selected[self.start] = 0.0
        # if self.goal is not None:
        #     self.selected[self.goal] = 1.01
        # self.path, self.n_neigh, self.neigh_graph, self.path_graph = (

        self.update_partial_sol()
