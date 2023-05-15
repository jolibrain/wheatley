import networkx as nx
from utils.graph_utils import all_longest_distances

class Rcpsp:
    def __init__(self, n_jobs, n_modes_per_job, successors, durations, resource_cons, resource_availabilities, n_renewable_resources, n_nonrenewable_resources=0, n_doubly_constrained_resources=0):
        # Number of jobs in the graph
        self.n_jobs = n_jobs
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
        self.n_resources = n_renewable_resources + n_nonrenewable_resources + n_doubly_constrained_resources
        # Durations for each job and for each mode. Durations are expressed through an array [MIN, MAX, MOD]
        self.durations = durations
        # Consumption for each job and for each mode of jobs
        self.resource_cons = resource_cons
        # Maximum capacity of each resource
        self.max_resource_availability = max(resource_availabilities)

        self.successors_id = [[] for j in range(n_jobs)]

        # Compute max resource consumption 
        self.max_resource_consumption=0
        
        for j in range(n_jobs):
            for k in range(n_modes_per_job[j]):
                for r in range(self.n_resources):
                    if self.max_resource_consumption<resource_cons[j][k][r]:
                        self.max_resource_consumption = resource_cons[j][k][r]
        

        self.predecessors = [[] for j in range(n_jobs)]
        self.predecessors_id = [[] for j in range(n_jobs)]
        
        for j in range(n_jobs):
            for succ in self.successors[j]:
                succ_id = self.job_to_id(succ)
                self.predecessors[succ_id].append(self.id_to_job(j))
                self.predecessors_id[succ_id].append(j)
                self.successors_id[j].append(succ_id) 
        
        self.createGraph()
        
    
    #Create graph with modes as nodes
    def createGraph(self):
        # Compute sources and sinks of the graph 
        sources_id = []
        sinks_id = []

        for j in range(self.n_jobs):
            if len(self.predecessors[j]) == 0:
                sources_id.append(j)
            if len(self.successors[j]) == 0:
                sinks_id.append(j)

        print("sources_id",sources_id)

        self.precGraph = nx.DiGraph()

        node_idx = 0
        self.__job_id_to_mode_id_2_node = [[] for j in range(self.n_jobs)]
        self.__node_to_job_id_mode_id = []
        for j in range(self.n_jobs):
            for mode_j in range(self.n_modes_per_job[j]):
                self.__job_id_to_mode_id_2_node[j].append(node_idx)
                self.__node_to_job_id_mode_id.append((j,mode_j))
                node_idx += 1

        for j in range(self.n_jobs):
            for succ in self.successors_id[j]:
                for mode_j in range(self.n_modes_per_job[j]):
                    for mode_succ in range(self.n_modes_per_job[succ]):
                        self.precGraph.add_edge(
                            self.__job_id_to_mode_id_2_node[j][mode_j], 
                            self.__job_id_to_mode_id_2_node[succ][mode_succ],
                            weight=self.durations[j][mode_j][0])

        
        # If only one source, use it as source for the graph. Otherwise add one source in the graph
        assert(len(sources_id)>0)
        source_id0 = sources_id[0]
        
        if len(sources_id)==1 \
            and self.n_modes_per_job[source_id0] == 1 \
            and self.durations[source_id0][0][0]==0 :
            self.source_id = source_id0
        else:
            self.source_id = node_idx
            node_idx += 1
            for s in sources_id:
                for m in range(self.n_modes_per_job[s]):
                    self.precGraph.add_edge(
                        self.source_id, 
                        self.__job_id_to_mode_id_2_node[s][m],
                        weight=0)

         # If only one sink, use it as sink for the graph. Otherwise add one sink in the graph
        assert(len(sinks_id)>0)
        sink_id0 = sinks_id[0]
        if len(sinks_id)==1 \
            and self.n_modes_per_job[sink_id0] == 1 \
            and self.durations[sink_id0][0][0]==0 :
            self.sink_id = sink_id0
        else:
            self.sink_id =  node_idx
            node_idx += 1
            for s in sinks_id:
                for m in range(self.n_modes_per_job[s]):
                    self.precGraph.add_edge(
                        self.__job_id_to_mode_id_2_node[s][m],
                        self.sink_id, 
                        weight=self.durations[s][m][0])
        

    def computeDistSourceStart(self):
        latest_dates = [0 for n in range(self.n_jobs)]

        dist_source = all_longest_distances(self.precGraph, self.source_id)
        print(dist_source)

        return dist_source

    def computeDistStartSink(self):
        dist_sink = all_longest_distances(self.precGraph, self.sink_id, reverse_graph=True)
        print(dist_sink)

        return dist_sink

    def job_to_id(self, j):
        return j-1
    
    def id_to_job(self, j):
        return j+1

    def mode_to_id(self, m):
        return m-1
    
    def id_to_mode(self, j):
        return m+1


    def sample(self, sampling_type):
        if sampling_type == "resource":
            return self.sample_resource()

    # samples the graph by randomly selecting n resources. Returns a new Rcpsp instance
    def sample_resource(self, n_resources_sample):
        if n_resources_sample >= self.n_resources:
            return self
        selected_resources = random.choices(range(n_resources),k=n_resources_sample)

    # samples the graph by taking all the jobs whose topological order is between min_order and max_order
    # TODO 
    def sample_topological(self, min_order, max_order):
        return self

    # samples the graph by taking all the jobs whose earliest start date id after start_horizon and latest start date is before end_horizon
    # TODO
    def sample_temporal(self, start_horizon, end_horizon):
        return self

    # samples the graph by taking randomly n_jobs in the graph
    # TODO
    def sample_n_jobs(self, n_jobs):
        return self
    
    
    

