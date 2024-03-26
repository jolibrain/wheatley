import numpy as np

# import matplotlib.pyplot as plt
import networkx as nx
import time
from psp.utils.utils import compute_resources_graph_np
from psp.utils.resource_timeline import ResourceTimeline
from psp.utils.resource_flowgraph import ResourceFlowGraph
from psp.solution import Solution

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.pyplot import cm
import io
import cv2
import torch


class State:
    # TODO
    def __init__(
        self,
        env_specification,
        problem_description,
        problem,
        deterministic=True,
        observe_conflicts_as_cliques=True,
        resource_model="flowGraph",  # or timeline
        normalize_features=True,
    ):
        self.problem = problem
        self.problem_description = problem_description
        self.n_features = env_specification.n_features
        if isinstance(self.problem, dict):
            self.n_nodes = self.problem["n_modes"]
        else:
            self.n_nodes = self.problem.n_modes
        self.features = np.zeros((self.n_nodes, self.n_features), dtype=float)
        self.deterministic = deterministic
        self.observe_conflicts_as_cliques = observe_conflicts_as_cliques
        self.env_specification = env_specification
        self.resource_model = resource_model
        if self.resource_model == "flowGraph":
            self.resourceModel = ResourceFlowGraph
        else:
            self.resourceModel = ResourceTimeline

        self.normalize = normalize_features

        self.add_rp_edges = env_specification.add_rp_edges
        self.factored_rp = env_specification.factored_rp

        self.step_horizon = env_specification.observation_horizon_step
        self.time_horizon = env_specification.observation_horizon_time

        self.edge_index = {}

        self.resource_prec_edges = []
        self.resource_prec_att = []
        self.remove_old_resource_info = env_specification.remove_old_resource_info

        self.remove_past_prec = env_specification.remove_past_prec

        # features :
        # 0: is_affected
        # 1: is selectable
        # 2: job_id (in case of several mode per job)
        # 3 : type (source/sink/none)
        # 4,5,6: durations (min max mode)
        # 7,8,9 : tct (min max mode)
        # 10.. 9+max_n_resources : level of resource i used by this mode (normalized)

        self.job_modes = []
        if isinstance(problem, dict):
            job_info = problem["job_info"]
            m = 0
            for n, j in enumerate(job_info):
                self.job_modes.append(list(range(m, m + j[0])))
                for mi in range(m, m + j[0]):
                    self.features[mi, 2] = n
                m += j[0]
        else:
            m = 0
            for n, j in enumerate(problem.n_modes_per_job):
                self.job_modes.append(list(range(m, m + j)))
                for mi in range(m, m + j):
                    self.features[mi, 2] = n
                m += j

        self.problem_edges = []
        if isinstance(problem, dict):
            for n, j in enumerate(job_info):
                for succ_job in j[1]:
                    for orig_mode in self.job_modes[n]:
                        for dest_mode in self.job_modes[succ_job - 1]:
                            self.problem_edges.append((orig_mode, dest_mode))
        else:
            for n, j in enumerate(problem.successors):
                for succ_job in j:
                    for orig_mode in self.job_modes[n]:
                        for dest_mode in self.job_modes[succ_job - 1]:
                            self.problem_edges.append((orig_mode, dest_mode))

        self.problem_graph = nx.DiGraph()
        if isinstance(problem, dict):
            self.problem_graph.add_nodes_from(range(0, problem["n_modes"]))
        else:
            self.problem_graph.add_nodes_from(range(0, problem.n_modes))
        self.problem_graph.add_edges_from(self.problem_edges)

        # nx.draw_networkx(self.graph)
        # plt.show()
        self.numpy_problem_graph = np.transpose(np.array(self.problem_edges))

        self.real_durations = None
        self.real_tct = np.zeros((self.n_nodes), dtype=float)

        self.reset_durations()
        self.reset_type()
        self.reset_tct()
        self.reset_is_affected()
        self.reset_resources()
        self.reset_selectable()

        if self.observe_conflicts_as_cliques:
            (
                self.resource_conf_edges,
                self.resource_conf_id,
                self.resource_conf_val,
                self.resource_conf_val_r,
            ) = compute_resources_graph_np(self.features[:, 10:])

        else:
            self.resource_conf_edges = None
            self.resource_conf_id = None
            self.resource_conf_val = None
            self.resource_conf_val_r = None

        # self.resources_edges.append((prec, succ))
        # self.resources_edges_att.append((on_start, critical))

    ########################### RESET/ INIT STUFF #############################

    def draw_real_durations(self, durs):
        if self.deterministic:
            return durs[:, 0]
        else:
            d = np.zeros((durs.shape[0]))
            null_durations = np.where(durs[1:-1, 1] == durs[1:-1, 2])
            fixed_durs = durs[1:-1, 2].copy()
            fixed_durs[null_durations] = 1e-6

            r = np.random.triangular(durs[1:-1, 1], durs[1:-1, 0], fixed_durs)

            # reinstating null durations
            durs[1:-1, 2][null_durations] = 0
            # r = np.random.triangular(durs[1:-1, 1], durs[1:-1, 0], durs[1:-1, 2])
            d[1:-1] = r
            return d

    def reset_durations(self, redraw_real=True):
        # at init, draw real durations from distrib if not deterministic
        if isinstance(self.problem, dict):
            for i in range(3):
                flat_dur = [
                    item for sublist in self.problem["durations"][i] for item in sublist
                ]
                self.features[:, 4 + i] = np.array(flat_dur)
        else:
            for i in range(3):
                flat_dur = [
                    item for sublist in self.problem.durations[i] for item in sublist
                ]
                self.features[:, 4 + i] = np.array(flat_dur)

        if redraw_real:
            self.real_durations = self.draw_real_durations(self.features[:, 4:7])
        self.max_duration = max(flat_dur)
        self.duration_upper_bound = 0
        for j in self.job_modes:
            self.duration_upper_bound += np.max(self.features[j, 6])
        self.undoable_makespan = self.duration_upper_bound + self.max_duration

    def reset_is_affected(self):
        self.features[:, 0] = 0

    def reset(self):
        self.reset_type()
        self.reset_tct()
        self.reset_is_affected()
        self.reset_resources()
        self.reset_selectable()
        self.edge_index = {}
        self.resource_prec_att = []
        self.resource_prec_edges = []

        if self.observe_conflicts_as_cliques:
            (
                self.resource_conf_edges,
                self.resource_conf_id,
                self.resource_conf_val,
                self.resource_conf_val_r,
            ) = compute_resources_graph_np(self.features[:, 10:])

        else:
            self.resource_conf_edges = None
            self.resource_conf_id = None
            self.resource_conf_val = None
            self.resource_conf_val_r = None

    def reset_frontier(self):
        self.nodes_in_frontier = set()
        for r in self.resources:
            for i in range(1, 4):
                for fe in r[i].frontier:
                    self.nodes_in_frontier.add(fe[1])

    def reset_resources(self):
        if isinstance(self.problem, dict):
            self.n_resources = self.problem["n_resources"]
            flat_res = np.array(
                [item for sublist in self.problem["resources"] for item in sublist],
                dtype=float,
            )
            # normalize resource usage
            for i in range(self.problem["n_resources"]):
                flat_res[:, i] /= self.problem["resource_availability"][i]
            self.resource_levels = np.array(self.problem["resource_availability"])
            self.features[:, 10 : 10 + self.n_resources] = flat_res

            self.resources = []
            for r in range(self.problem["n_renewable_resources"]):
                self.resources.append([])
                for i in range(4):
                    self.resources[r].append(
                        self.resourceModel(
                            max_level=1.0,
                            unit_val=1.0 / self.resource_levels[r],
                            renewable=True,
                        )
                    )

            for r in range(
                self.problem["n_renewable_resources"],
                self.problem["n_renewable_resources"]
                + self.problem["n_nonrenewable_resources"],
            ):
                self.resources.append([])
                for i in range(4):
                    self.resources[r].append(
                        self.resourceModel(
                            max_level=1.0,
                            unit_val=1.0 / self.resource_levels[r],
                            renewable=False,
                        )
                    )
        else:
            self.n_resources = self.problem.n_resources
            flat_res = np.array(
                [item for sublist in self.problem.resource_cons for item in sublist],
                dtype=float,
            )
            # normalize resource usage
            for i in range(self.problem.n_resources):
                flat_res[:, i] /= self.problem.resource_availabilities[i]
            self.resource_levels = np.array(self.problem.resource_availabilities)
            self.features[:, 10 : 10 + self.n_resources] = flat_res

            self.resources = []
            for r in range(self.problem.n_renewable_resources):
                self.resources.append([])
                for i in range(4):
                    self.resources[r].append(
                        self.resourceModel(
                            max_level=1.0,
                            unit_val=1.0 / self.resource_levels[r],
                            renewable=True,
                        )
                    )

            for r in range(
                self.problem.n_renewable_resources,
                self.problem.n_renewable_resources
                + self.problem.n_nonrenewable_resources,
            ):
                self.resources.append([])
                for i in range(4):
                    self.resources[r].append(
                        self.resourceModel(
                            max_level=1.0,
                            unit_val=1.0 / self.resource_levels[r],
                            renewable=False,
                        )
                    )

        assert len(self.resources) == self.n_resources
        self.reset_frontier()

    def reset_type(self):
        self.features[:, 3] = 0
        in_deg = np.array(self.problem_graph.in_degree())

        no_parents_pos = np.where(in_deg[:, 1] == 0)[0]
        no_parents = in_deg[no_parents_pos][:, 0]

        self.features[no_parents, 3] = -1  # source

        out_deg = np.array(self.problem_graph.out_degree())
        no_child_pos = np.where(out_deg[:, 1] == 0)[0]
        no_child = out_deg[no_child_pos][:, 0]
        self.features[no_child, 3] = 1  # sink

    def get_weight(self, source, dest, edge_attr):
        if source >= self.features.shape[0]:
            # case of virtual added source
            return 0
        return self.features[source, 4]

    def reset_selectable(self):
        self.features[:, 1] = 0
        in_deg = np.array(self.problem_graph.in_degree())

        no_parents_pos = np.where(in_deg[:, 1] == 0)[0]
        no_parents = in_deg[no_parents_pos][:, 0]

        self.features[no_parents, 1] = 1

    def reset_tct(self):
        self.update_completion_times(None)
        sinks = np.where(self.types() == 1)[0]
        sinks_makespans = self.tct(sinks)
        self.init_makespan = np.max(sinks_makespans)

    ############################### ACCESSORS ############################

    def tct(self, nodeid):
        return self.features[nodeid, 7:10]

    def all_tct_real(self):
        return self.real_tct

    def tct_real(self, nodeid):
        return self.real_tct[nodeid]

    def set_tct(self, nodeid, ct):
        self.features[nodeid, 7:10] = ct

    def set_tct_real(self, nodeid, ct):
        self.real_tct[nodeid] = ct

    def all_durations(self):
        return self.features[:, 4:7]

    def durations(self, nodeid):
        return self.features[nodeid, 4:7]

    def duration_real(self, nodeid):
        return self.real_durations[nodeid]

    def all_duration_real(self):
        return self.real_durations[:]

    def resources_usage(self, node_id):
        return self.features[node_id, 10:]

    def all_resources_usage(self):
        return self.features[:, 10:]

    def resource_usage(self, node_id, r):
        return self.features[node_id, 10 + r]

    def remove_res(self, node_ids):
        self.features[node_ids, 10:] = 0

    def selectables(self):
        return self.features[:, 1]

    def types(self):
        return self.features[:, 3]

    def selectable(self, node_id):
        return self.features[node_id, 1] == 1

    def set_unselectable(self, node_id):
        self.features[node_id, 1] = 0

    def set_selectable(self, node_id):
        self.features[node_id, 1] = 1

    def jobid(self, node_id):
        return self.features[node_id, 2].astype(int)

    def all_jobid(self):
        return self.features[:, 2].astype(int)

    def modes(self, job_id):
        return self.job_modes[job_id]

    def set_affected(self, nodeid):
        self.features[nodeid, 0] = 1

    def affected(self, nodeid):
        return self.features[nodeid, 0] == 1

    def all_affected(self):
        return self.features[:, 0] == 1

    def all_not_affected(self):
        return self.features[:, 0] == 0

    def trivial_actions(self):
        return np.where(
            np.logical_and(
                self.selectables() == 1, np.all(self.all_resources_usage() == 0, axis=1)
            )
        )[0]

    def unmasked_actions(self):
        return np.where(self.selectables() == 1)[0]

    ############################### EXTERNAL API ############################

    def done(self):
        return self.finished() or self.succeeded()
        # return np.sum(self.features[:, 0]) == self.problem["n_jobs"]

    def finished(self):
        return np.where(self.selectables() == 1)[0].size == 0

    def succeeded(self):
        sinks = np.where(self.types() == 1)[0]
        # sinks_selected = self.features[sinks, 0] == 1
        all_sinks_selected = np.all(self.features[sinks, 0] == 1)
        # return self.features[-1, 0] == 1
        return all_sinks_selected

    def to_features_and_edge_index(self, normalize):
        if self.observe_conflicts_as_cliques:
            rce = self.resource_conf_edges
            rca = np.stack(
                [
                    self.resource_conf_id,
                    self.resource_conf_val,
                    self.resource_conf_val_r,
                ],
                axis=1,
            )
        else:
            rce = None
            rca = None

        if self.add_rp_edges != "none":
            if len(self.resource_prec_edges) > 0:
                if self.add_rp_edges == "all":
                    rpe = np.transpose(self.resource_prec_edges)
                    rpa = np.concatenate(self.resource_prec_att)
                elif self.add_rp_edges == "frontier":
                    rpe, rpa = self.rp_edges_to_keep()
                elif self.add_rp_edges == "frontier_strict":
                    rpe, rpa = self.rp_edges_to_keep_strict()
                else:
                    raise NotImplementedError
            else:
                rpe = None
                rpa = None
            return (
                self.process_features(),
                self.numpy_problem_graph,
                rce,
                rca,
                rpe,
                rpa,
            )

        if self.remove_past_prec:
            fresh_nodes = list(
                set(np.where(self.all_not_affected())[0]).union(self.nodes_in_frontier)
            )

            return (
                self.process_features(),
                self.remove_past_edges(fresh_nodes),
                rce,
                rca,
            )
        return (
            self.process_features(),
            self.numpy_problem_graph,
            rce,
            rca,
        )

    def affect_job(self, node_id):
        self.affect_node(node_id)
        self.compute_dates_on_affectation(node_id)
        self.mask_wrt_non_renewable_resources()
        self.update_completion_times_after(node_id)
        if self.remove_old_resource_info:
            self.remove_res_frontier()
        else:
            self.nodes_in_frontier = set()
            for r in self.resources:
                for i in range(1, 4):
                    for fe in r[i].frontier:
                        self.nodes_in_frontier.add(fe[1])

    def mask_wrt_non_renewable_resources(self):
        if isinstance(self.problem, dict):
            if self.problem["n_nonrenewable_resources"] == 0:
                return
        else:
            if self.problem.n_nonrenewable_resources == 0:
                return
        selectables = np.where(self.features[:, 1] == 1)[0]
        # TODO rewrite
        for n in selectables:
            for r, level in enumerate(
                self.resources_usage(n)[self.problem["n_renewable_resources"], :]
            ):
                if level == 0:
                    continue
                if not self.resources[r][0].still_available(level):
                    self.set_unselectable(n)
                    break

    def render_fail(self):
        plt.text(
            0.5,
            0.5,
            "invalid",
            size=50,
            rotation=0.0,
            ha="center",
            va="center",
            bbox=dict(
                boxstyle="round",
                ec=(1.0, 0.5, 0.5),
                fc=(1.0, 0.8, 0.8),
            ),
        )
        figimg = io.BytesIO()
        plt.savefig(figimg, format="png", dpi=150)
        plt.clf()
        plt.close("all")
        figimg.seek(0)
        npimg = np.fromstring(figimg.read(), dtype="uint8")
        cvimg = cv2.imdecode(npimg, cv2.IMREAD_UNCHANGED)
        npimg = np.transpose(cvimg, (2, 0, 1))
        torchimg = torch.from_numpy(npimg)
        return torchimg

    def render_solution(self, schedule, scaling=1.0):
        starts = [int(i * scaling) for i in schedule[0]]
        modes = schedule[1]
        if isinstance(self.problem, dict):
            n_jobs = self.problem["n_jobs"]
            nres = self.problem["n_resources"]
            maxres = self.problem["resource_availability"]
            rusage = [self.problem["resources"][j][modes[j]] for j in range(n_jobs)]
        else:
            n_jobs = self.problem.n_jobs
            nres = self.problem.n_resources
            maxres = self.problem.resource_availabilities
            rusage = [self.problem.resource_cons[j][modes[j]] for j in range(n_jobs)]

        ends = [
            # starts[j] + self.problem["durations"][0][j][modes[j]] for j in range(n_jobs)
            starts[j] + int(self.duration_real(self.job_modes[j][modes[j]]))
            for j in range(n_jobs)
        ]

        levels = []
        for r in range(nres):
            levels.append([0] * int((max(ends) + 1)))

        # color = cm.gnuplot2(np.linspace(0, 1, n_jobs))
        color = cm.rainbow(np.linspace(0, 1, len(starts)))

        fig, ax = plt.subplots(nres)
        for i in range(nres):
            ax[i].set_xlim([-1, max(ends) * 1.2])
            ax[i].set_ylim([0, maxres[i]])
            ax[i].set_ylabel(f"R {i+1}")

        patches = []

        for i in range(n_jobs):
            for r in range(nres):
                rect = Rectangle(
                    (starts[i], levels[r][starts[i]]),
                    ends[i] - starts[i],
                    rusage[i][r],
                    edgecolor=color[i],
                    facecolor=color[i],
                    fill=True,
                    alpha=0.2,
                    lw=1,
                    label=f"J{i}/m{modes[i]}",
                )
                ax[r].add_patch(rect)
                if rusage[i][r] != 0:
                    max_level = max(levels[r][starts[i] : ends[i] + 1])
                    ax[r].text(
                        starts[i] + (ends[i] - starts[i]) / 2,
                        max_level + rusage[i][r] / 2 - 0.2,
                        str(i),
                        color=color[i],
                    )
                for t in range(starts[i], ends[i]):
                    levels[r][t] += rusage[i][r]
                if r == 0:
                    patches.append(rect)

        ax[nres - 1].set_xlabel("time")
        fig.tight_layout(pad=2)
        fig.legend(handles=patches)

        figimg = io.BytesIO()
        fig.savefig(figimg, format="png", dpi=150)
        plt.clf()
        plt.close("all")
        figimg.seek(0)
        npimg = np.fromstring(figimg.read(), dtype="uint8")
        cvimg = cv2.imdecode(npimg, cv2.IMREAD_UNCHANGED)
        npimg = np.transpose(cvimg, (2, 0, 1))
        torchimg = torch.from_numpy(npimg)
        return torchimg

    def get_solution(self):
        if not self.succeeded():
            return None
        tct = self.real_tct
        schedule = tct - self.real_durations[:]

        return Solution.from_mode_schedule(
            schedule,
            self.problem,
            self.all_affected(),
            self.all_jobid(),
            real_durations=self.real_durations,
        )

    ############################ INTERNAL ###############################

    def affect_node(self, nodeid):
        # should be selectable
        assert self.selectable(nodeid)
        # all modes become unselectable
        self.set_unselectable(self.modes(self.jobid(nodeid)))
        if self.remove_old_resource_info:
            other_modes = self.modes(self.jobid(nodeid)).copy()
            other_modes.remove(nodeid)
            self.remove_res(other_modes)
        # mark as affected
        self.set_affected(nodeid)
        # make sucessor selectable, if other parents *jobs* are affected
        for successor in self.problem_graph.successors(nodeid):
            parents_jobs = set(
                [self.jobid(pm) for pm in self.problem_graph.predecessors(successor)]
            )
            # no need to test job from currently affected node
            parents_jobs.remove(self.features[nodeid, 2])
            # check if one mode per job is affected
            all_parent_jobs_affected = True
            for pj in parents_jobs:
                pjm = self.modes(pj)
                job_is_affected = False
                for pjmm in pjm:
                    if self.affected(pjmm):
                        job_is_affected = True
                        break
                if not job_is_affected:
                    all_parent_jobs_affected = False
                    break
            if all_parent_jobs_affected:
                # make selectable, at last
                self.set_selectable(successor)

    def process_features(self):
        if self.normalize or self.step_horizon != 0 or self.time_horizon != 0:
            feat = np.copy(self.features)
        else:
            feat = self.features
        if self.normalize:
            feat[:, 4:10] /= self.max_duration
        if self.step_horizon != 0 or self.time_horizon != 0:
            # recompute reachable nodes in step_horizon steps in not_affected ones
            sources = np.where(self.features[:, 1] == 1)[0]
            v_source = self.problem_graph.number_of_nodes()
            self.problem_graph.add_node(v_source)
            for i in sources:
                self.problem_graph.add_edge(v_source, i)
            # not_affected = (np.where(self.features[:, 0] == 0)[0]).tolist()
            to_remove = []
            if self.step_horizon != 0:
                step_depth = nx.shortest_path_length(
                    self.problem_graph, source=v_source
                )
                for i in step_depth.keys():
                    if i == v_source:
                        continue
                    if step_depth[i] > self.step_horizon:
                        to_remove.append(i)
            if self.time_horizon != 0:
                time_depth = nx.shortest_path_length(
                    self.problem_graph, source=v_source, weight=self.get_weight
                )
                for i in time_depth.keys():
                    if i == v_source:
                        continue
                    if time_depth[i] > self.time_horizon:
                        to_remove.append(i)

            feat[to_remove, 10:] = 0
            self.problem_graph.remove_node(v_source)

        return feat

    def get_last_finishing_dates(self, jobs):
        if len(jobs) == 0:
            return np.zeros((3))
        return np.amax(self.tct(jobs), axis=0)

    def get_last_finishing_dates_real(self, jobs):
        if len(jobs) == 0:
            return np.array([0])
        return np.array([np.amax(self.tct_real(jobs), axis=0)])

    def compute_dates_on_affectation(self, node_id):
        job_parents = list(self.problem_graph.predecessors(node_id))
        affected_parents = np.array(job_parents)[
            np.where(self.affected(job_parents))[0]
        ]
        last_parent_finish_date = self.get_last_finishing_dates(affected_parents)
        last_parent_finish_date_real = self.get_last_finishing_dates_real(
            affected_parents
        )

        resources_used = np.where(self.resources_usage(node_id) != 0)[0]
        max_r_date = np.array([0.0, 0.0, 0.0, 0.0])
        constraining_resource = [None] * 4

        # TODO : in case on non-renewable resources, we may fail to find resources
        if self.resource_model == "timeline":
            pred_on_resource = [None] * 4
            pred_on_resource_is_start = [None] * 4
            if self.deterministic:
                indices_to_add = [3]
            else:
                indices_to_add = [1, 2, 3]
            for r in resources_used:
                rad = self.resource_available_date(r, self.resource_usage(node_id, r))
                # rad is 3-long list of (date, jobid, start_tp)
                for i in range(4):
                    if rad[i][0] > max_r_date[i]:
                        max_r_date[i] = rad[i][0]
                        pred_on_resource[i] = rad[i][1]
                        pred_on_resource_is_start[i] = rad[i][2]
                        constraining_resource[i] = r
                        if pred_on_resource[i] is not None and i in indices_to_add:
                            self.add_resource_precedence(
                                pred_on_resource[i],
                                node_id,
                                pred_on_resource_is_start[i],
                                True,
                                i,
                                r,
                            )
                    else:
                        if pred_on_resource[i] is not None and i in indices_to_add:
                            self.add_resource_precedence(
                                pred_on_resource[i],
                                node_id,
                                pred_on_resource_is_start[i],
                                False,
                                i,
                                r,
                            )
        else:
            for r in resources_used:
                rad = self.resource_available_date_flowgraph(
                    r, self.resource_usage(node_id, r)
                )
                # rad is only the date
                for i in range(4):
                    if rad[i] > max_r_date[i]:
                        max_r_date[i] = rad[i]
                        constraining_resource[i] = r

        # do a min per coord
        start = np.maximum(
            max_r_date,
            np.concatenate([last_parent_finish_date_real, last_parent_finish_date]),
        )
        self.set_tct(node_id, start[1:] + self.durations(node_id))
        self.set_tct_real(node_id, start[0] + self.duration_real(node_id))

        self.consume(0, node_id, start[0], self.tct_real(node_id))
        for i in range(3):
            self.consume(i + 1, node_id, start[i + 1], self.tct(node_id)[i])

        if self.resource_model == "flowGraph" and self.add_rp_edges != "none":
            # extract graph info
            self.update_resource_prec(constraining_resource)

    def update_resource_prec(self, constraining_resource):
        if self.factored_rp:
            for r in range(self.n_resources):
                for i in range(3):
                    for ie, e in enumerate(self.resources[r][i + 1].new_edges_cache):
                        if e not in self.edge_index:
                            self.resource_prec_att.append(
                                np.zeros(
                                    (1, self.env_specification.max_n_resources * 3)
                                )
                            )
                            self.edge_index[e] = len(self.resource_prec_att) - 1
                            self.resource_prec_edges.append(e)
                        self.resource_prec_att[self.edge_index[e]][
                            0, r * 3 + i
                        ] = self.resources[r][i + 1].new_edges_att_cache[ie]
                    self.resources[r][i + 1].reset_new_cache()

        else:
            for r in range(self.n_resources):
                if len(self.resources[r][1].new_edges_cache) != 0:
                    for i in range(3):
                        self.resource_prec_edges.extend(
                            self.resources[r][i + 1].new_edges_cache
                        )
                        rpa = np.empty(
                            (len(self.resources[r][i + 1].new_edges_att_cache), 4)
                        )
                        rpa[:, 0] = r
                        rpa[:, 1] = self.resources[r][i + 1].new_edges_att_cache
                        rpa[:, 2] = constraining_resource[i] == r
                        rpa[:, 3] = i
                        self.resource_prec_att.append(rpa)
                        self.resources[r][i + 1].reset_new_cache()

    def add_resource_precedence(self, prec, succ, on_start, critical, timetype, rid):
        self.resource_prec_edges.append((prec, succ))
        self.resource_prec_att.append(
            (rid, self.resource_usage(succ, rid), critical, timetype)
        )

    def resource_available_date(self, rid, level):
        return [self.resources[rid][i].availability(level) for i in range(4)]

    def resource_available_date_flowgraph(self, rid, level):
        return [self.resources[rid][i].availability(level) for i in range(4)]

    def consume(self, timeindex, node_id, start, end):
        resources_used = np.where(self.resources_usage(node_id) != 0)[0]
        for r in resources_used:
            self.resources[r][timeindex].consume(
                node_id,
                self.resource_usage(node_id, r),
                start,
                end,
                debug=(timeindex == 0),
            )

    def update_completion_times_after(self, node_id):
        for n in self.problem_graph.successors(node_id):
            self.update_completion_times(n)

    def update_completion_times(self, node_id):
        if node_id is None:
            open_nodes = np.where(np.array(self.problem_graph.in_degree())[:, 1] == 0)[
                0
            ].tolist()
        else:
            open_nodes = [node_id]
        while open_nodes:
            cur_node_id = open_nodes.pop(0)

            if self.problem_graph.in_degree(cur_node_id) == 0:
                max_tct_predecessors = np.zeros((3))
                max_tct_predecessors_real = 0
            else:
                task_comp_time_pred = np.stack(
                    [self.tct(p) for p in self.problem_graph.predecessors(cur_node_id)]
                )
                max_tct_predecessors = np.max(task_comp_time_pred, 0)
                max_tct_predecessors_real = max(
                    [
                        self.tct_real(p)
                        for p in self.problem_graph.predecessors(cur_node_id)
                    ]
                )

            new_completion_time = max_tct_predecessors + self.durations(cur_node_id)
            new_completion_time_real = max_tct_predecessors_real + self.duration_real(
                cur_node_id
            )

            if (
                (node_id is None)
                or (new_completion_time_real != self.tct_real(cur_node_id))
                or (np.any(np.not_equal(new_completion_time, self.tct(cur_node_id))))
            ):
                self.set_tct(cur_node_id, new_completion_time)
                self.set_tct_real(cur_node_id, new_completion_time_real)
                for successor in self.problem_graph.successors(cur_node_id):
                    to_open = True
                    for p in self.problem_graph.predecessors(successor):
                        if p in open_nodes:
                            to_open = False
                            break
                    if to_open:
                        open_nodes.append(successor)

    def remove_res_frontier(self):
        new_nodes_in_frontier = set()
        for r in self.resources:
            for i in range(1, 4):
                for fe in r[i].frontier:
                    new_nodes_in_frontier.add(fe[1])

        removed_from_frontier = self.nodes_in_frontier - new_nodes_in_frontier
        self.remove_res(list(removed_from_frontier))
        self.nodes_in_frontier = new_nodes_in_frontier

    def remove_past_edges(self, fresh_nodes):
        new_edges = []
        for edge in self.problem_edges:
            if edge[0] in fresh_nodes or edge[1] in fresh_nodes:
                new_edges.append((edge[0], edge[1]))
        return np.transpose(np.array(new_edges))

    def rp_edges_to_keep(self):
        new_edges = []
        edge_indices = []
        for ei, edge in enumerate(self.resource_prec_edges):
            if edge[0] in self.nodes_in_frontier or edge[1] in self.nodes_in_frontier:
                new_edges.append(edge)
                edge_indices.append(ei)
        if len(new_edges) == 0:
            return None, None
        return (
            np.transpose(new_edges),
            np.concatenate(self.resource_prec_att)[edge_indices],
        )

    def rp_edges_to_keep_strict(self):
        new_edges = []
        edge_indices = []
        for ei, edge in enumerate(self.resource_prec_edges):
            if edge[0] in self.nodes_in_frontier and edge[1] in self.nodes_in_frontier:
                new_edges.append(edge)
                edge_indices.append(ei)
        if len(new_edges) == 0:
            return None, None
        return (
            np.transpose(new_edges),
            np.concatenate(self.resource_prec_att)[edge_indices],
        )
