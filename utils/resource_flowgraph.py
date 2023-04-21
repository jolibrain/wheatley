# This class allows to model cumulative resources. For a resource with a capacity capa_r, it ensures that at any time t,
# the sum of consumptions of r is lower than or equal to capa_r.
# Careful, such a resource type is not always suitable for representing a set of disjunctive resources.


class ResourceFlowGraph:
    def __init__(self, max_level, unit_val=1.0, renewable=True):
        # max_level should be the true capacity
        self.max_level = max_level
        self.renewable = renewable
        # sourceNode contains the id of the source Node in the graph with all activities
        self.sourceNode = 0
        # contains all nodes of graph. A node is simply the id of the activity
        self.nodes = [0]
        # edges are labeled by the flow between each node
        self.edges = []
        self.edges_att = []
        # the frontier contains a list of tuples [releaseDate,nodeId,consReleased],
        # that indicate that consReleased units of the resource are available at releaseDate from nodeId
        self.frontier = []
        self.unit_val = unit_val
        for i in range(0, int(self.max_level / self.unit_val)):
            self.frontier.insert(i, [0, 0, self.unit_val])

    def availability(self, level):
        # should return date, position in frontier associated to date
        # indicating if tp is start or end of previous consumer
        # this version return first available date
        assert level <= self.max_level

        # starts from the beginning of the frontier and returns the smallest date d_i such that \sum_{j <= i} consReleased[j] >= level
        available_cons = 0
        frontier_idx = -1
        while available_cons < level:
            frontier_idx = frontier_idx + 1
            available_cons = available_cons + self.frontier[frontier_idx][2]
        return self.frontier[frontier_idx][0]

    # Returns the max position pos in self.frontier such that frontier[pos][0] <= date
    # if date > frontier[-1][0], returns len(frontier)
    # if date < fronter[0][0], return -1
    def find_max_pos(self, date):
        if len(self.frontier) == 0:
            return 0
        # assert date >= self.frontier[0][0]
        if date < self.frontier[0][0]:
            return -1
        pos = 0
        while (pos < len(self.frontier)) and (self.frontier[pos][0] <= date):
            pos = pos + 1
        return pos - 1

    # inserts after the last position pos such that frontier[pos][0] <= date
    def insert_in_frontier(self, date, consumer_id, level):
        max_pos = self.find_max_pos(date)
        for i in range(int(level / self.unit_val)):
            self.frontier.insert(max_pos + 1, [date, consumer_id, self.unit_val])

    def consume(self, consumer_id, level, start, end, debug=False):
        assert level <= self.max_level
        cur_pos = self.find_max_pos(start)
        available_capa = 0
        to_add_in_frontier = []
        flow_dict = {}
        while available_capa < level:
            available_capa = available_capa + self.frontier[cur_pos][2]
            to_add_in_frontier.append([end, consumer_id, self.unit_val])
            origin_node = self.frontier[cur_pos][1]
            self.frontier.pop(cur_pos)
            cur_pos = cur_pos - 1
            if origin_node in flow_dict:
                flow_dict[origin_node] = flow_dict[origin_node] + self.unit_val
            else:
                flow_dict[origin_node] = self.unit_val
        for t in to_add_in_frontier:
            self.insert_in_frontier(t[0], t[1], t[2])
        self.nodes.append(consumer_id)
        for node in flow_dict:
            self.edges.append((node, consumer_id))
            self.edges_att.append(flow_dict[node])

    def generate_graph(self):
        return 0
