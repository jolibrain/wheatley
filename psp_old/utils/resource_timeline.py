class ResourceTimeline:
    def __init__(self, max_level, renewable=True, allow_before_last=True):
        self.max_level = max_level
        self.renewable = renewable
        self.timepoints = [[0, None, max_level, None]]
        # better rep : ordered list of events for every date
        # ?
        # self.timepoints = [(0, [[None, max_level, None]])]
        self.global_availability = [0, max_level]
        # if allow_before_last : a consumer that consumers after an aready consuming one can start
        # to consume before already consuming ones
        # if not, allow to consume start to consume only as early as previously consuming tasks
        self.allow_before_last = allow_before_last

    def availability(self, level):
        # should return date, previous consumer, boolean
        # indicating if tp is start or end of previous consumer
        # IMPORTANT NOTE:
        # this version return first available date, even if before previously inserted consumers

        ret_tp_index = len(self.timepoints) - 1
        while ret_tp_index > 0:
            cand_tp_index = ret_tp_index - 1
            # find timepoint with different previous date
            while cand_tp_index > 0 and (
                self.timepoints[ret_tp_index][0] == self.timepoints[cand_tp_index][0]
            ):
                cand_tp_index -= 1
            if self.timepoints[cand_tp_index][2] < level:
                break
            # go back in time
            ret_tp_index = cand_tp_index
            if not self.allow_before_last and self.timepoints[cand_tp_index][3]:
                # if before last is off, do not allow to get back before start of other consumers
                break
        return (
            self.timepoints[ret_tp_index][0],
            self.timepoints[ret_tp_index][1],
            self.timepoints[ret_tp_index][3],
        )

    def find_pos(self, date):
        for i in range(len(self.timepoints)):
            if date >= self.timepoints[i][0]:
                if i == len(self.timepoints) - 1 or date < self.timepoints[i + 1][0]:
                    return i + 1
        return None

    def consume(self, consumer_id, level, start, end):
        start_pos = self.find_pos(start)
        level_before = self.timepoints[start_pos - 1][2]
        self.timepoints.insert(start_pos, [start, consumer_id, level_before, True])
        for i in range(start_pos, len(self.timepoints)):
            self.timepoints[i][2] -= level
            assert self.timepoints[i][2] >= 0

        if self.renewable:
            end_pos = self.find_pos(end)
            level_before_end = self.timepoints[end_pos - 1][2]
            self.timepoints.insert(end_pos, [end, consumer_id, level_before_end, False])
            for i in range(end_pos, len(self.timepoints)):
                self.timepoints[i][2] += level
                assert self.timepoints[i][2] <= self.max_level

    def global_availability(self):
        avail = []
        previous_tp = self.timepoints[0]
        for i in range(1, len(self.timepoints)):
            if not self.timepoints[i][0] == previous_tp[0]:
                avail.append([previous_tp[0], previous_tp[2]])
            previous_tp = self.timepoints[i]
        return avail
