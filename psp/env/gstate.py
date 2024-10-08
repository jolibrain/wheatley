import numpy as np

# import matplotlib.pyplot as plt
import time
from psp.utils.utils import compute_resources_graph_torch
from psp.utils.resource_timeline import ResourceTimeline
from psp.utils.resource_flowgraph import ResourceFlowGraph
from psp.solution import Solution

from concurrent.futures import ThreadPoolExecutor

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.pyplot import cm
import io
import cv2
import torch
import bisect
from psp.graph.graph_factory import GraphFactory
from psp.graph.graph import Graph


class GState:
    def __init__(
        self,
        env_specification,
        problem_description,
        problem,
        deterministic=True,
        observe_conflicts_as_cliques=True,
        normalize_features=True,
        pyg=True,
    ):
        self.pyg = pyg
        # self.tpe = ThreadPoolExecutor()
        self.problem = problem
        self.problem_description = problem_description
        self.n_features = env_specification.n_features
        # self.device = torch.device("cuda:2")
        self.device = torch.device("cpu")
        if isinstance(self.problem, dict):
            self.n_nodes = self.problem["n_modes"]
        else:
            self.n_nodes = self.problem.n_modes

        # self.features = np.zeros((self.n_nodes, self.n_features), dtype=float)
        self.deterministic = deterministic
        self.observe_conflicts_as_cliques = observe_conflicts_as_cliques
        self.env_specification = env_specification
        self.resourceModel = ResourceFlowGraph

        self.normalize = normalize_features

        self.add_rp_edges = env_specification.add_rp_edges
        self.factored_rp = env_specification.factored_rp

        self.edge_index = {}

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
                m += j[0]
        else:
            m = 0
            for n, j in enumerate(problem.n_modes_per_job):
                self.job_modes.append(list(range(m, m + j)))
                m += j

        self.problem_edges = []
        if isinstance(problem, dict):
            for n, j in enumerate(job_info):
                for succ_job in j[1]:
                    for orig_mode in self.job_modes[n]:
                        for dest_mode in self.job_modes[succ_job - 1]:
                            self.problem_edges.append((orig_mode, dest_mode))
        else:
            for n, j in enumerate(problem.successors_id):
                for succ_job in j:
                    for orig_mode in self.job_modes[n]:
                        for dest_mode in self.job_modes[succ_job]:
                            self.problem_edges.append((orig_mode, dest_mode))

        # self.problem_graph = nx.DiGraph(self.problem_edges)
        # nx.draw_networkx(self.graph)
        # plt.show()
        # self.numpy_problem_graph = np.transpose(np.array(self.problem_edges))

        self.reset_graph()
        self.reset_fresh_nodes()
        self.reset_durations()
        self.reset_tct()
        self.reset_is_affected()
        self.reset_resources()
        self.reset_selectable()
        self.reset_type()
        self.reset_conflicts_as_cliques()

        # self.resources_edges.append((prec, succ))
        # self.resources_edges_att.append((on_start, critical))

    ########################### RESET/ INIT STUFF #############################

    def draw_real_durations(self, durs):
        if self.deterministic:
            return durs[:, 0]
        else:
            d = torch.empty(durs.shape[0])
            zero_length = torch.where(durs[:, 1] == durs[:, 2])[0]
            nonzero_length = torch.where(durs[:, 1] != durs[:, 2])[0]
            d[nonzero_length] = torch.tensor(
                np.random.triangular(
                    durs[nonzero_length, 1],
                    durs[nonzero_length, 0],
                    durs[nonzero_length, 2],
                ),
                dtype=torch.float,
            )
            d[zero_length] = durs[zero_length, 0]
            return d

    def reset_fresh_nodes(self):
        self.fresh_nodes = list(range(self.n_nodes))

    def reset_conflicts_as_cliques(self):
        if self.observe_conflicts_as_cliques:
            (
                resource_conf_edges,
                resource_conf_id,
                resource_conf_val,
                resource_conf_val_r,
            ) = compute_resources_graph_torch(self.graph.ndata("resources"))
            self.graph.add_edges(
                resource_conf_edges[0],
                resource_conf_edges[1],
                data={
                    "rid": resource_conf_id,
                    "val": resource_conf_val,
                    "valr": resource_conf_val_r,
                },
                etype="rc",
            )

    def reset_graph(self):
        self.graph = GraphFactory.create_graph(
            self.problem_edges,
            self.n_nodes,
            self.factored_rp,
            self.observe_conflicts_as_cliques,
            self.device,
            pyg=self.pyg,
        )

        m = 0
        for n, j in enumerate(self.problem.n_modes_per_job):
            for mi in range(m, m + j):
                self.graph.ndata("job")[mi] = n
            m += j

    def reset_durations(self, redraw_real=True):
        # at init, draw real durations from distrib if not deterministic
        for i in range(3):
            flat_dur = [
                item for sublist in self.problem.durations[i] for item in sublist
            ]
            self.graph.ndata("durations")[:, i] = torch.tensor(flat_dur).to(self.device)

        if redraw_real:
            # self.real_durations = self.draw_real_durations(self.features[:, 4:7])
            self.real_durations = self.draw_real_durations(
                self.graph.ndata("durations")
            )
        self.max_duration = max(flat_dur)
        if self.normalize:
            self.graph.set_ndata(
                "normalized_durations",
                self.graph.ndata("durations") / self.max_duration,
            )

        self.duration_upper_bound = 0
        for j in self.job_modes:
            self.duration_upper_bound += torch.max(self.graph.ndata("durations")[j, 2])
        self.undoable_makespan = self.duration_upper_bound + self.max_duration

    def reset_is_affected(self):
        self.graph.set_ndata(
            "affected",
            torch.zeros((self.n_nodes), dtype=torch.float, device=self.device),
        )
        self.graph.set_ndata(
            "past", torch.zeros((self.n_nodes), dtype=torch.bool, device=self.device)
        )

    def reset(self):
        self.reset_graph()
        self.reset_fresh_nodes()
        self.reset_durations(redraw_real=False)
        self.reset_tct()
        self.reset_is_affected()
        self.reset_resources()
        self.reset_selectable()
        self.reset_type()
        self.reset_conflicts_as_cliques()

    def reset_frontier(self):
        self.nodes_in_frontier = set()
        for r in self.resources:
            for i in range(1, 4):
                for fe in r[i].frontier:
                    self.nodes_in_frontier.add(fe[1])

    def reset_resources(self):
        self.n_resources = self.problem.n_resources
        self.graph.set_ndata(
            "resources",
            torch.zeros(
                (self.n_nodes, self.n_resources),
                dtype=torch.float,
                device=self.device,
            ),
        )
        flat_res = torch.tensor(
            [item for sublist in self.problem.resource_cons for item in sublist],
            dtype=torch.float,
        )
        # normalize resource usage
        for i in range(self.problem.n_resources):
            flat_res[:, i] /= self.problem.resource_availabilities[i]
        self.resource_levels = torch.tensor(self.problem.resource_availabilities).to(
            self.device
        )

        # self.features[:, 10 : 10 + self.n_resources] = flat_res
        self.graph.ndata("resources")[:] = flat_res

        self.resources = []
        for r in range(self.problem.n_renewable_resources):
            self.resources.append([])
            for i in range(4):
                self.resources[r].append(
                    self.resourceModel(
                        max_level=1.0,
                        unit_val=1.0 / self.resource_levels[r].item(),
                        renewable=True,
                    )
                )

        for r in range(
            self.problem.n_renewable_resources,
            self.problem.n_renewable_resources + self.problem.n_nonrenewable_resources,
        ):
            self.resources.append([])
            for i in range(4):
                self.resources[r].append(
                    self.resourceModel(
                        max_level=1.0,
                        unit_val=1.0 / self.resource_levels[r].item(),
                        renewable=False,
                    )
                )

        self.res_cal = []
        if self.problem.res_cal is not None:
            for r in range(
                self.problem.n_renewable_resources
                + self.problem.n_nonrenewable_resources
            ):
                self.res_cal.append(self.problem.cals[self.problem.res_cal[r]])
            self.res_cal_id = [
                list(self.problem.cals.keys()).index(c) for c in self.problem.res_cal
            ]
        else:
            self.res_cal_id = [0] * self.n_resources

        self.graph.set_global_data("res_cal", torch.tensor(self.res_cal_id))

        assert len(self.resources) == self.n_resources
        self.reset_frontier()

    def reset_type(self):
        self.graph.set_ndata(
            "type", torch.where(self.graph.in_degrees() == 0, -1.0, 0.0)
        )  # source

        self.graph.set_ndata(
            "type",
            torch.where(
                self.graph.out_degrees() == 0,
                1.0,
                self.graph.ndata("type"),
            ),
        )  # sink

    def reset_selectable(self):
        self.graph.set_ndata(
            "selectable",
            torch.where(self.graph.in_degrees() == 0, True, False),
        )  # source

    def reset_tct(self):
        self.real_tct = torch.zeros(
            (self.n_nodes), dtype=torch.float, device=self.device
        )
        self.graph.set_ndata(
            "tct", torch.zeros((self.n_nodes, 3), dtype=torch.float, device=self.device)
        )
        self.update_completion_times(None)

    ############################### ACCESSORS ############################

    def tct(self, nodeid):
        return self.graph.ndata("tct")[nodeid]

    def all_tct(self):
        return self.graph.ndata("tct")

    def all_tct_real(self):
        return self.real_tct

    def tct_real(self, nodeid):
        return self.real_tct[nodeid]

    def set_tct(self, nodeid, ct):
        self.graph.ndata("tct")[nodeid] = ct

    def set_tct_real(self, nodeid, ct):
        self.real_tct[nodeid] = ct

    def all_durations(self):
        return self.graph.ndata("durations")

    def durations(self, nodeid):
        return self.graph.ndata("durations")[nodeid]

    def duration_real(self, nodeid):
        return self.real_durations[nodeid]

    def all_duration_real(self):
        return self.real_durations[:]

    def resources_usage(self, node_id):
        return self.graph.ndata("resources")[node_id]

    def all_resources_usage(self):
        return self.graph.ndata("resources")

    def resource_usage(self, node_id, r):
        return self.graph.ndata("resources")[node_id][r]

    def remove_res(self, node_ids):
        self.graph.ndata("resources")[node_ids] = 0.0

    def selectables(self):
        return self.graph.ndata("selectable")

    def types(self):
        return self.graph.ndata("type")

    def selectable(self, node_id):
        return self.graph.ndata("selectable")[node_id]

    def set_unselectable(self, node_id):
        self.graph.ndata("selectable")[node_id] = False

    def set_selectable(self, node_id):
        self.graph.ndata("selectable")[node_id] = True

    def jobid(self, node_id):
        return self.graph.ndata("job")[node_id]

    def all_jobid(self):
        return self.graph.ndata("job")

    def modes(self, job_id):
        return self.job_modes[job_id]

    def set_affected(self, nodeid):
        self.graph.ndata("affected")[nodeid] = 1

    def affected(self, nodeid):
        return self.graph.ndata("affected")[nodeid] == 1

    def all_affected(self):
        return self.graph.ndata("affected") == 1

    def all_not_affected(self):
        return self.graph.ndata("affected") == 0

    def set_past(self, nodeid):
        self.graph.ndata("past")[nodeid] = True

    def get_pasts(self):
        return self.graph.ndata("past")

    def trivial_actions(self):
        return torch.where(
            torch.logical_and(
                self.selectables(),
                torch.all(self.all_resources_usage() == 0, 1),
            )
        )[0]

    def unmasked_actions(self):
        return self.selectables().to(torch.device("cpu")).numpy()

    ############################### EXTERNAL API ############################

    def done(self):
        return self.finished() or self.succeeded()

    def finished(self):
        # return torch.where(self.selectables())[0].size()[0] == 0
        return torch.all(torch.logical_not(self.selectables()))

    def succeeded(self):
        sinks = torch.where(self.types() == 1)[0]
        # sinks_selected = self.features[sinks, 0] == 1
        all_sinks_selected = torch.all(self.graph.ndata("affected")[sinks] == 1)
        # return self.features[-1, 0] == 1
        return all_sinks_selected

    def observe(self):
        self.normalize_features()
        return self.graph

    def affect_job(self, node_id):
        self.affect_node(node_id)
        self.compute_dates_on_affectation(node_id)
        self.mask_wrt_non_renewable_resources()
        self.update_completion_times_after(node_id)
        if (
            self.remove_old_resource_info
            or self.add_rp_edges != "none"
            or self.remove_past_prec
        ):
            nodes_removed_from_frontier = self.update_frontier()
        if self.remove_old_resource_info:
            self.remove_res_frontier(nodes_removed_from_frontier)
        if self.add_rp_edges == "frontier":
            self.remove_rp_edges(strict=False)
        elif self.add_rp_edges == "frontier_strict":
            self.remove_rp_edges(strict=True)
        if self.remove_past_prec:
            self.remove_past_edges(nodes_removed_from_frontier, node_id)

    def mask_wrt_non_renewable_resources(self):
        if self.problem.n_nonrenewable_resources == 0:
            return
        # TODO : parallelize
        selectables = torch.where(self.selectables())[0]
        for n in selectables:
            for r, level in enumerate(self.resources_usage(n)):
                if not self.resources[r][0].still_available(level.item()):
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
            schedule.to(torch.device("cpu")),
            self.problem,
            self.all_affected().to(torch.device("cpu")),
            self.all_jobid().to(torch.device("cpu")),
            real_durations=self.real_durations.to(torch.device("cpu")),
        )

    ############################ INTERNAL ###############################

    def affect_node(self, nodeid):
        # should be selectable
        assert self.selectable(
            nodeid
        ), f"invalid action selected:  {nodeid} ; selectables: {self.selectables() == 1}"
        # all modes become unselectable
        self.set_past(self.modes(self.jobid(nodeid).item()))
        self.set_unselectable(self.modes(self.jobid(nodeid).item()))
        if self.remove_old_resource_info:
            other_modes = self.modes(self.jobid(nodeid)).copy()
            other_modes.remove(nodeid)
            self.remove_res(other_modes)
        # mark as affected
        self.set_affected(nodeid)
        # make sucessor selectable, if other parents *jobs* are affected
        # for successor in self.graph.successors(nodeid, etype="prec"):
        for successor in self.graph.successors(nodeid):
            # parents_jobs = set(
            #     [
            #         self.jobid(pm).item()
            #         # for pm in self.graph.predecessors(successor, etype="prec")
            #         for pm in self.graph.predecessors(successor.item())
            #     ]
            # )
            ####
            parents_jobs = set(
                self.jobid(self.graph.predecessors(successor.item())).tolist()
            )
            ####
            # no need to test job from currently affected node
            parents_jobs.remove(self.graph.ndata("job")[nodeid].item())
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

    def normalize_features(self):
        if self.normalize:
            self.graph.set_ndata(
                "normalized_tct", self.graph.ndata("tct") / self.max_duration
            )

    def get_last_finishing_dates(self, jobs):
        if len(jobs) == 0:
            return torch.zeros((3), dtype=torch.float, device=self.device)
        return torch.amax(self.tct(jobs), axis=0)

    def get_last_finishing_dates_real(self, jobs):
        if len(jobs) == 0:
            return torch.tensor([0.0], device=self.device)
        return torch.amax(self.tct_real(jobs), 0, keepdim=True)

    def compute_dates_on_affectation(self, node_id):
        # job_parents = self.graph.predecessors(node_id, etype="prec")
        job_parents = self.graph.predecessors(node_id)
        affected_parents = job_parents[torch.where(self.affected(job_parents))[0]]
        last_parent_finish_date = self.get_last_finishing_dates(affected_parents)
        last_parent_finish_date_real = self.get_last_finishing_dates_real(
            affected_parents
        )

        resources_used = torch.where(self.resources_usage(node_id) != 0)[0]
        max_r_date = torch.tensor(
            [0.0, 0.0, 0.0, 0.0], dtype=torch.float, device=self.device
        )
        constraining_resource = [None] * 4

        for r in resources_used:
            rad = self.resource_available_date_flowgraph(
                r.item(), self.resource_usage(node_id, r)
            )
            # rad is only the date
            for i in range(4):
                if rad[i] > max_r_date[i]:
                    max_r_date[i] = float(rad[i])
                    constraining_resource[i] = r.item()

        # do a min per coord
        start = torch.maximum(
            max_r_date,
            torch.cat([last_parent_finish_date_real, last_parent_finish_date]),
        )

        ends_with_cal, end_real_with_cal = self.compute_ends_with_cal(
            node_id, resources_used, start
        )
        self.set_tct(node_id, ends_with_cal)
        self.set_tct_real(node_id, end_real_with_cal)
        # self.set_tct(node_id, start[1:] + self.durations(node_id))
        # self.set_tct_real(node_id, start[0] + self.duration_real(node_id))

        self.consume(0, node_id, start[0], self.tct_real(node_id))
        for i in range(3):
            self.consume(i + 1, node_id, start[i + 1], self.tct(node_id)[i])

        if self.add_rp_edges != "none":
            # extract graph info
            self.update_resource_prec(constraining_resource)

    def compute_ends_with_cal(self, node_id, resources_used, start):
        durations = self.durations(node_id)
        duration_real = self.duration_real(node_id)
        starts = start[1:]
        start_real = start[0]
        end_dates = durations + starts
        end_date_real = duration_real + start_real

        if len(self.res_cal) == 0:
            return end_dates, end_date_real

        for r in resources_used:
            p = bisect.bisect(self.res_cal[r], start_real, key=lambda x: x[0]) - 1
            if p == -1:  # date is before first interval
                p = 0

            remaining_duration = duration_real.item()  # do not forget to copy !
            opening_duration = self.res_cal[r][p][1] - max(
                self.res_cal[r][p][0], start_real
            )

            while remaining_duration > opening_duration:
                remaining_duration -= opening_duration
                p += 1
                if p >= len(self.res_cal[r]):
                    print("!!!!! not enough openings for task")
                    print("time type: real")
                    print("resource: ", r)
                    print("calendar: ", self.res_cal[r])
                    print("node_id: ", node_id)
                    print("start time: ", start_real)
                    print("duration", duration_real)
                    print("remaining duration: ", remaining_duration)
                    print("trying p:", p)
                    exit()
                opening_duration = self.res_cal[r][p][1] - self.res_cal[r][p][0]

            local_end_date_real = (
                max(start_real.item(), self.res_cal[r][p][0]) + remaining_duration
            )

            if local_end_date_real > end_date_real:
                end_date_real = local_end_date_real

            for i in range(3):
                p = bisect.bisect(self.res_cal[r], starts[i], key=lambda x: x[0]) - 1
                if p == -1:
                    p = 0
                remaining_duration = durations[i].item()  # do not forget to copy !
                opening_duration = self.res_cal[r][p][1] - max(
                    self.res_cal[r][p][0], starts[i]
                )
                while remaining_duration > opening_duration:
                    remaining_duration -= opening_duration
                    p += 1
                    if p >= len(self.res_cal[r]):
                        print("!!!!! not enough openings for task")
                        print("time type: ", i)
                        print("resource: ", r)
                        print("calendar: ", self.res_cal[r])
                        print("node_id: ", node_id)
                        print("start time: ", starts[i])
                        print("duration", durations[i])
                        print("remaining duration: ", remaining_duration)
                        print("trying p:", p)
                        exit()
                    opening_duration = self.res_cal[r][p][1] - self.res_cal[r][p][0]
                local_end_date = (
                    max(starts[i], self.res_cal[r][p][0]) + remaining_duration
                )
                if local_end_date > end_dates[i]:
                    end_dates[i] = local_end_date

        return end_dates, end_date_real

    def update_resource_prec(self, constraining_resource):
        if self.factored_rp:
            for r in range(self.n_resources):
                for i in range(3):
                    for ie, e in enumerate(self.resources[r][i + 1].new_edges_cache):
                        if not self.graph.has_edges_between(e[0], e[1], etype="rp"):
                            self.graph.add_edges(
                                e[0],
                                e[1],
                                etype="rp",
                                data={
                                    "r": torch.zeros(
                                        (self.env_specification.max_n_resources * 3)
                                    )
                                },
                            )
                        self.graph.edata["r"][("n", "rp", "n")][r * 3 + i] = (
                            self.resources[r][i + 1].new_edges_att_cache[ie]
                        )
                    self.resources[r][i + 1].reset_new_cache()

        else:
            for r in range(self.n_resources):
                for i in range(3):
                    if self.resources[r][i + 1].new_edges_cache:
                        ne = (
                            torch.tensor(self.resources[r][i + 1].new_edges_cache)
                            .t()
                            .to(self.device)
                        )
                        rpa = torch.empty(
                            (len(self.resources[r][i + 1].new_edges_cache), 4),
                            dtype=torch.float,
                        )
                        rpa[:, 0] = r
                        rpa[:, 1] = torch.tensor(
                            self.resources[r][i + 1].new_edges_att_cache
                        )
                        rpa[:, 2] = int(constraining_resource[i] == r)
                        rpa[:, 3] = i

                        self.graph.add_edges(ne[0], ne[1], etype="rp", data={"r": rpa})
                    self.resources[r][i + 1].reset_new_cache()

    def resource_available_date(self, rid, level):
        return [self.resources[rid][i].availability(level.item()) for i in range(4)]

    def resource_available_date_flowgraph(self, rid, level):
        return [self.resources[rid][i].availability(level.item()) for i in range(4)]

    def consume(self, timeindex, node_id, start, end):
        resources_used = torch.where(self.resources_usage(node_id) != 0)[0]
        for r in resources_used:
            self.resources[r][timeindex].consume(
                node_id,
                self.resource_usage(node_id, r).item(),
                start.item(),
                end.item(),
                debug=(timeindex == 0),
            )

    def update_completion_times_after(self, node_id):
        # for n in self.graph.successors(node_id, etype="prec"):
        for n in self.graph.successors(node_id):
            self.update_completion_times(n)

    def max_thread_safe(self, cur_node_id):
        return torch.max(
            # self.tct(self.graph.predecessors(cur_node_id, etype="prec")), 0, True
            self.tct(self.graph.predecessors(cur_node_id)),
            0,
            True,
        )

    def update_completion_times(self, node_id):
        initial_tct = False
        if node_id is None:
            indeg = self.graph.in_degrees()
            open_nodes = torch.where(indeg == 0)[0].tolist()
            initial_tct = True
        else:
            open_nodes = [node_id.item()]
        while open_nodes:
            cur_node_id = open_nodes.pop(0)

            # if self.graph.in_degrees(cur_node_id, etype="prec") == 0:
            if self.graph.indeg(cur_node_id) == 0:
                max_tct_predecessors = torch.zeros(
                    (3), dtype=torch.float, device=self.device
                )
                max_tct_predecessors_real = torch.zeros(
                    (1), dtype=torch.float, device=self.device
                )
            else:
                # V0
                # max_tct_predecessors = torch.max(
                #     self.tct(self.graph.predecessors(cur_node_id)),
                #     0,
                #     keepdim=True,
                # )[0]
                # V1 old workaroud (not needed anymore)
                # max_tct_predecessors = self.tpe.submit(
                #     self.max_thread_safe, cur_node_id
                # ).result()[0]
                # preds = self.graph.predecessors(cur_node_id, etype="prec")

                # V2 old workaround (not needed anymore)
                preds = self.graph.predecessors(cur_node_id)
                max_tct_predecessors = torch.from_numpy(
                    np.max(
                        self.tct(preds).to(torch.device("cpu")).numpy(),
                        axis=0,
                        keepdims=True,
                    )
                ).to(self.device)

                # V0
                # max_tct_predecessors_real = torch.max(
                #     self.tct_real(self.graph.predecessors(cur_node_id))
                # )
                # V2 old workaroud (not needed anymore)
                # preds = self.graph.predecessors(cur_node_id)
                max_tct_predecessors_real = torch.tensor(
                    np.max(self.tct_real(preds).to(torch.device("cpu")).numpy())
                ).to(self.device)

            new_completion_time = max_tct_predecessors + self.durations(cur_node_id)
            new_completion_time_real = max_tct_predecessors_real + self.duration_real(
                cur_node_id
            )

            # not quicker ...
            # self.set_tct(cur_node_id, new_completion_time)
            # self.set_tct_real(cur_node_id, new_completion_time_real)

            # if initial_tct:
            #     sucs = self.graph.successors(cur_node_id).tolist()
            #     open_nodes = [n for n in open_nodes if n not in sucs]
            #     open_nodes.extend(sucs)
            #     continue

            # sucs = self.graph.successors(cur_node_id)
            # real_starts = self.tct_real(sucs) - self.duration_real(sucs)
            # gt_real = torch.gt(new_completion_time_real, real_starts)
            # starts = self.tct(sucs) - self.durations(sucs)
            # gt_all = torch.gt(new_completion_time, starts)
            # if torch.any(gt_real) or torch.any(gt_all):
            #     sucs_real = set(torch.where(gt_real)[0].tolist())
            #     sucs_all = set(torch.where(gt_all)[0].tolist())
            #     sucs = list(sucs_real.union(sucs_all))
            #     open_nodes = [n for n in open_nodes if n not in sucs]
            #     open_nodes.extend(sucs)

            if (
                initial_tct
                or (torch.gt(new_completion_time_real, self.tct_real(cur_node_id)))
                or (torch.any(torch.gt(new_completion_time, self.tct(cur_node_id))))
            ):
                self.set_tct(cur_node_id, new_completion_time)
                self.set_tct_real(cur_node_id, new_completion_time_real)

                sucs = self.graph.successors(cur_node_id).tolist()
                open_nodes = [n for n in open_nodes if n not in sucs]
                open_nodes.extend(sucs)

    def update_frontier(self):
        new_nodes_in_frontier = set()
        for r in self.resources:
            for i in range(1, 4):
                for fe in r[i].frontier:
                    new_nodes_in_frontier.add(fe[1])
        removed_from_frontier = self.nodes_in_frontier - new_nodes_in_frontier
        self.nodes_in_frontier = new_nodes_in_frontier
        return removed_from_frontier

    def remove_rp_edges(self, strict):
        # keep only edges with an end into the frontie
        # remove edges with no end in the frontier
        rp_edges = self.graph.edges("rp")
        nif = torch.tensor(list(self.nodes_in_frontier), dtype=torch.long)
        c1 = torch.eq(rp_edges[0].unsqueeze(1), nif.unsqueeze(0))
        c2 = torch.eq(rp_edges[1].unsqueeze(1), nif.unsqueeze(0))
        # src_not_in_frontier = torch.logical_not(torch.any(c1, dim=1))
        # dst_not_in_frontier = torch.logical_not(torch.any(c2, dim=1))
        if strict:
            c3 = torch.logical_or(c1, c2)
        else:
            c3 = torch.logical_and(c1, c2)
        to_remove = torch.where(c3)[0]
        self.graph.remove_edges(rp_edges[2][to_remove], etype="rp")

    def remove_res_frontier(self, nodes_removed_from_frontier):
        self.remove_res(list(nodes_removed_from_frontier))
        if self.observe_conflicts_as_cliques:
            rc_edges = self.graph.edges(etype="rc")
            nrrf = torch.tensor(list(nodes_removed_from_frontier))
            c1 = torch.eq(rc_edges[0].unsqueeze(1), nrrf.unsqueeze(0))
            c2 = torch.eq(rc_edges[1].unsqueeze(1), nrrf.unsqueeze(0))
            # src_in_removed = torch.any(c1, dim=1)
            # dst_in_removed = torch.any(c2, dim=1)
            to_remove = torch.where(torch.logical_or(c1, c2))[0]
            eids = rc_edges[2][to_remove]
            self.graph.remove_edges(eids, etype="rc")

    def remove_past_edges(self, removed_from_frontier, newly_affected):
        nodes_to_remove = removed_from_frontier
        if not newly_affected in self.nodes_in_frontier:
            nodes_to_remove.add(newly_affected)
        pr_edges = self.graph.edges(etype="prec")
        nitr = torch.tensor(list(nodes_to_remove), dtype=torch.long)
        c = torch.eq(pr_edges[1].unsqueeze(1), nitr.unsqueeze(0))
        dst_in_nitr = torch.any(c, dim=1)
        to_remove = torch.where(dst_in_nitr)[0]
        self.graph.remove_edges(pr_edges[2][to_remove], etype="prec")

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
