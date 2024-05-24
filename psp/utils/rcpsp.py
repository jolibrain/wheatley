import networkx as nx
import numpy

from .graph_utils import all_longest_distances


# Allows to model a RCPSP.
class Rcpsp:
    def __init__(
        self,
        pb_id,
        job_labels,
        n_modes_per_job,
        successors,
        durations,
        resource_cons,
        resource_availabilities,
        n_renewable_resources,
        n_nonrenewable_resources=0,
        n_doubly_constrained_resources=0,
        due_dates=None,
        res_cal=None,
        cals=None,
        display_trivial=False,
    ):
        self.pb_id = pb_id

        # job id (generally an int, but not necessarily)
        self.job_labels = job_labels
        # Number of modes for each job
        self.n_modes_per_job = n_modes_per_job
        # Total number of modes in the RCPSP
        self.n_modes = sum(n_modes_per_job)
        # Successors for each job. Successors are given by a list
        self.successors = successors
        # Capacity of each resource
        self.resource_availabilities = resource_availabilities
        # Number of renewable resources
        self.n_renewable_resources = n_renewable_resources
        # Number of non renewable resources
        self.n_nonrenewable_resources = n_nonrenewable_resources
        # Number of doubly constrained resources
        self.n_doubly_constrained_resources = n_doubly_constrained_resources
        # Total number of resources
        self.n_resources = (
            n_renewable_resources
            + n_nonrenewable_resources
            + n_doubly_constrained_resources
        )
        # Durations for each job and for each mode. Durations can be expressed through an array [MIN, MAX, MOD]
        self.durations = durations
        # Consumption for each job and for each mode of jobs
        self.resource_cons = resource_cons
        if display_trivial:
            n_trivial = 0
            for rc in resource_cons:
                all_mode_dont_consume = True
                for mode_cons in rc:
                    if not all(v == 0 for v in mode_cons):
                        all_mode_dont_consume = False
                        break
                if all_mode_dont_consume:
                    n_trivial += 1
            print(
                "Problem ",
                pb_id,
                " has ",
                n_trivial,
                "/",
                len(resource_cons),
                "trivial actions",
            )

        # resources calendars :
        # if of calendar for each ressource
        self.res_cal = res_cal
        # calendar intervals (opening)
        self.cals = cals

        # due dates, for forthcoming tardiness
        self.due_dates = due_dates

        # Maximum capacity of each resource
        self.max_resource_availability = max(resource_availabilities)

        # Compute max resource consumption
        self.max_resource_consumption = 0
        self.add_source_sink_if_needed()

        for j in range(len(self.job_labels)):
            for k in range(self.n_modes_per_job[j]):
                for r in range(self.n_resources):
                    if self.max_resource_consumption < self.resource_cons[j][k][r]:
                        self.max_resource_consumption = self.resource_cons[j][k][r]

        self.predecessors = [[] for j in range(len(self.job_labels))]
        self.predecessors_id = [[] for j in range(len(self.job_labels))]

        # Successors of each node using id of tasks (starting from 0)
        self.successors_id = [[] for j in range(len(self.job_labels))]
        for j in range(len(self.job_labels)):
            for succ in self.successors[j]:
                succ_id = self.job_to_id(succ)
                self.predecessors[succ_id].append(self.id_to_job(j))
                self.predecessors_id[succ_id].append(j)
                self.successors_id[j].append(succ_id)

        self.createGraph()

        self.n_jobs = len(self.job_labels)

    def add_source_sink_if_needed(self):
        n_parents = {}
        n_children = {}
        for jl in self.job_labels:
            n_parents[jl] = 0
            n_children[jl] = 0
        for n, s in enumerate(self.successors):
            n_children[self.id_to_job(n)] += len(s)
            for c in s:
                n_parents[c] += 1

        no_parents = []
        no_children = []
        for j in self.job_labels:
            if n_parents[j] == 0:
                no_parents.append(j)
            if n_children[j] == 0:
                no_children.append(j)

        if len(no_parents) != 1 or self.job_to_id(no_parents[0]) != 0:
            print("adding source")
            # job_labels
            self.job_labels.insert(0, "source")
            # successors
            self.successors.insert(0, no_parents)
            # n_modes_per_job
            self.n_modes_per_job.insert(0, 1)
            # n_modes
            self.n_modes += 1
            # durations
            for i in range(3):
                self.durations[i].insert(0, [0.0])
            # resource_cons
            self.resource_cons.insert(0, [[0] * self.n_resources])
            # due_dates
            self.due_dates.insert(0, None)

        if (
            len(no_children) != 1
            or self.job_to_id(no_children[0]) != len(self.job_labels) - 1
        ):
            print("adding sink")
            # job_labels
            self.job_labels.append("sink")
            # successors
            for s in self.successors:
                if len(s) == 0:
                    s.append("sink")
            self.successors.append([])
            # n_modes_per_job
            self.n_modes_per_job.append(1)
            # n_modes
            self.n_modes += 1
            # durations
            for i in range(3):
                self.durations[i].append([0.0])
            # resource_cons
            self.resource_cons.append([[0] * self.n_resources])
            # due_dates
            self.due_dates.append(None)

    # Create graph with modes as nodes
    def createGraph(self):
        # Compute sources and sinks of the graph
        sources_id = []
        sinks_id = []

        for j in range(len(self.job_labels)):
            if len(self.predecessors[j]) == 0:
                sources_id.append(j)
            if len(self.successors[j]) == 0:
                sinks_id.append(j)

        # print("sources_id",sources_id)

        self.precGraph = nx.DiGraph()

        node_idx = 0
        self.__job_id_to_mode_id_2_node = [[] for j in range(len(self.job_labels))]
        self.__node_to_job_id_mode_id = []
        for j in range(len(self.job_labels)):
            for mode_j in range(self.n_modes_per_job[j]):
                self.__job_id_to_mode_id_2_node[j].append(node_idx)
                self.__node_to_job_id_mode_id.append((j, mode_j))
                node_idx += 1

        # print("job_id_to_mode_to_node",self.__job_id_to_mode_id_2_node)

        for j in range(len(self.job_labels)):
            for succ in self.successors_id[j]:
                for mode_j in range(self.n_modes_per_job[j]):
                    for mode_succ in range(self.n_modes_per_job[succ]):
                        self.precGraph.add_edge(
                            self.__job_id_to_mode_id_2_node[j][mode_j],
                            self.__job_id_to_mode_id_2_node[succ][mode_succ],
                            weight=self.durations[0][j][mode_j],
                        )
                        # print("add_edge", self.__job_id_to_mode_id_2_node[j][mode_j]+1,
                        # self.__job_id_to_mode_id_2_node[succ][mode_succ]+1,
                        # self.durations[j][mode_j][0])

        # If only one source, use it as source for the graph. Otherwise add one source in the graph
        assert len(sources_id) > 0
        # source_id0 = sources_id[0]

        # if (
        #     len(sources_id) == 1
        #     and self.n_modes_per_job[source_id0] == 1
        #     and self.durations[source_id0][0][0] == 0
        # ):
        #     self.source_id = source_id0
        # else:
        #     self.source_id = node_idx
        #     node_idx += 1
        #     for s in sources_id:
        #         for m in range(self.n_modes_per_job[s]):
        #             self.precGraph.add_edge(
        #                 self.source_id, self.__job_id_to_mode_id_2_node[s][m], weight=0
        #             )
        #     self.successors_id.append(sources_id)
        #     self.successors.append([self.id_to_job(id) for id in sources_id])

        # If only one sink, use it as sink for the graph. Otherwise add one sink in the graph
        assert len(sinks_id) > 0
        # sink_id0 = sinks_id[0]
        # if (
        #     len(sinks_id) == 1
        #     and self.n_modes_per_job[sink_id0] == 1
        #     and self.durations[sink_id0][0][0] == 0
        #     and sinks_id[0] != self.source_id
        # ):
        #     self.sink_id = sink_id0
        # else:
        #     self.sink_id = node_idx
        #     node_idx += 1
        #     for s in sinks_id:
        #         for m in range(self.n_modes_per_job[s]):
        #             self.precGraph.add_edge(
        #                 self.__job_id_to_mode_id_2_node[s][m],
        #                 self.sink_id,
        #                 weight=self.durations[s][m][0],
        #             )
        #         self.successors_id[s].append(self.sink_id)
        #         self.successors[self.id_to_job(s)].append(self.id_to_job(self.sink_id))
        #     self.successors_id.append([])
        #     self.successors.append([])

    def computeDistSourceStart(self):
        dist_source = all_longest_distances(self.precGraph, self.source_id)
        # print(dist_source)

        return dist_source

    def computeDistStartSink(self):
        dist_sink = all_longest_distances(
            self.precGraph, self.sink_id, reverse_graph=True
        )
        # print(dist_sink)

        return dist_sink

    def job_to_id(self, j):
        return self.job_labels.index(j)

    def id_to_job(self, j):
        return self.job_labels[j]

    def sample(self, sampling_type):
        if sampling_type == "resource":
            return self.sample_resource()

    # samples the graph by randomly selecting n resources. Returns a new Rcpsp instance
    def sample_resource(self, n_resources_sample):
        if n_resources_sample >= self.n_resources:
            return self
        selected_resources = random.choices(range(n_resources), k=n_resources_sample)

    # samples the graph by taking all the jobs whose topological order is between min_order (>0) and max_order exclusive
    # TODO
    def sample_topological(self, min_order, max_order):
        print("min_order", min_order, "max_order", max_order)
        generations = list(nx.topological_generations(self.precGraph))
        ranks = len(generations)
        max_order = min(max_order, ranks)
        new_generations = generations[min_order:max_order]

        if min_order > 0:
            new_generations.insert(0, [self.source_id])
        if max_order < ranks:
            new_generations.append([self.sink_id])

        # print(new_generations)

        s_jobs = []
        for g in new_generations:
            s_jobs.extend(g)
        print(s_jobs)

        s_job_to_id = {}
        next_id = 0
        for j in s_jobs:
            s_job_to_id[j] = next_id
            next_id += 1

        print("map job to id", s_job_to_id)

        s_n_jobs = len(s_jobs)
        s_n_modes_per_job = [self.n_modes_per_job[j] for j in s_jobs]
        # print(s_n_modes_per_job)

        # Retrieves successors of nodes in new generations
        s_successors = []
        for j in s_jobs:
            s_successors.append(
                [s_job_to_id[succ] for succ in self.successors_id[j] if succ in s_jobs]
            )

        # Adds succs between source and nodes in the first selected generation
        if len(new_generations) > 1 and min_order > 1:
            s_successors[0].extend([s_job_to_id[j] for j in new_generations[1]])

        # Adds precs between nodes in the last selected generation and sink
        if len(new_generations) > 2 and max_order < ranks - 1:
            for j in new_generations[-2]:
                s_successors[s_job_to_id[j]].append(s_job_to_id[self.sink_id])

        print(s_successors)

        s_durations = []
        s_resource_cons = []
        for j in s_jobs:
            s_durations.append(self.durations[j])
            s_resource_cons.append(self.resource_cons[j])

        print(s_durations)

        return Rcpsp(
            s_n_jobs,
            s_n_modes_per_job,
            s_successors,
            s_durations,
            s_resource_cons,
            self.resource_availabilities,
            self.n_renewable_resources,
            n_nonrenewable_resources=self.n_nonrenewable_resources,
            n_doubly_constrained_resources=self.n_doubly_constrained_resources,
        )

    # samples the graph by taking all the jobs whose earliest start date id after start_horizon and latest start date is before end_horizon
    # TODO
    def sample_temporal(self, start_horizon, end_horizon):
        return self

    # samples the graph by taking randomly n_jobs in the graph
    # TODO
    def sample_n_jobs(self, n_jobs):
        return self

    def __eq__(self, obj):
        if isinstance(obj, dict):
            return False
        if self.n_modes_per_job != obj.n_modes_per_job:
            return False
        if self.n_modes != obj.n_modes:
            return False
        if self.successors != obj.successors:
            return False
        if self.resource_availabilities != obj.resource_availabilities:
            return False
        if self.n_renewable_resources != obj.n_renewable_resources:
            return False
        if self.n_nonrenewable_resources != obj.n_nonrenewable_resources:
            return False
        if self.n_doubly_constrained_resources != obj.n_doubly_constrained_resources:
            return False
        if self.n_resources != obj.n_resources:
            return False
        if self.durations != obj.durations:
            return False
        if self.resource_cons != obj.resource_cons:
            return False
        if self.max_resource_availability != obj.max_resource_availability:
            return False
        if self.successors_id != obj.successors_id:
            return False
        if self.max_resource_consumption != obj.max_resource_consumption:
            return False
        if self.predecessors != obj.predecessors:
            return False
        if self.predecessors_id != obj.predecessors_id:
            return False
        return True
