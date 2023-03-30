import numpy as np


def load_problem(
    problem_file,
    taillard_offset=False,
    deterministic=True,
    load_from_job=0,
    load_max_jobs=-1,
    generate_bounds=-1.0,
):
    # Customized problem loader
    # - support for bounded duration uncertainty
    # - support for unattributed machines
    # - support for columns < number of machines

    print("generate_bounds=", generate_bounds)

    if not deterministic:
        print("Loading problem with uncertainties, using customized format")
        if generate_bounds > 0:
            print("Generating random duration bounds of ", generate_bounds, " %")

    with open(problem_file, "r") as f:
        line = next(f)
        while line[0] == "#":
            line = next(f)

        # header
        header = line
        head_list = [int(i) for i in header.split()]
        assert len(head_list) == 2
        n_j = head_list[0]
        n_m = head_list[1]

        line = next(f)
        while line[0] == "#":
            line = next(f)

        # matrix of durations
        np_lines = []
        for j in range(n_j):
            dur_list = []
            for i in line.split():
                add_dur = float(i)
                if add_dur == 0:
                    add_dur = 0.1
                elif add_dur < 0:
                    add_dur = -1.0
                dur_list.append(add_dur)
            while len(dur_list) < n_m:
                dur_list.append(-1.0)
            np_lines.append(np.array(dur_list))
            line = next(f)
        durations = np.stack(np_lines)

        if deterministic:
            durations = np.expand_dims(durations, axis=2)
            durations = np.repeat(durations, 4, axis=2)
        elif generate_bounds > 0.0:
            mode_durations = durations
            min_durations = np.subtract(
                durations,
                generate_bounds * durations,
                out=durations.copy(),
                where=durations != -1,
            )
            max_durations = np.add(
                durations,
                generate_bounds * durations,
                out=durations.copy(),
                where=durations != -1,
            )
            real_durations = np.zeros((n_j, n_m)) - 1
            durations = np.stack(
                [real_durations, min_durations, max_durations, mode_durations], axis=2
            )
            # sys.exit()
        else:
            mode_durations = durations

            while line[0] == "#":
                line = next(f)

            np_lines = []
            for j in range(n_j):
                dur_list = []
                for i in line.split():
                    add_dur = float(i)
                    if add_dur == 0:
                        add_dur = 0.1
                    elif add_dur < 0:
                        add_dur = -1.0
                    dur_list.append(add_dur)
                while len(dur_list) < n_m:
                    dur_list.append(-1.0)
                np_lines.append(np.array(dur_list))
                line = next(f)
            min_durations = np.stack(np_lines)

            while line[0] == "#":
                line = next(f)

            np_lines = []
            for j in range(n_j):
                dur_list = []
                for i in line.split():
                    add_dur = float(i)
                    if add_dur == 0:
                        add_dur = 0.1
                    elif add_dur < 0:
                        add_dur = -1.0
                    dur_list.append(add_dur)
                while len(dur_list) < n_m:
                    dur_list.append(-1.0)
                np_lines.append(np.array(dur_list))
                line = next(f)
            max_durations = np.stack(np_lines)

            real_durations = np.zeros((n_j, n_m)) - 1

            durations = np.stack(
                [real_durations, min_durations, max_durations, mode_durations], axis=2
            )

        while line[0] == "#":
            line = next(f)

        # matrix of affectations
        if taillard_offset:
            toffset = 1
        else:
            toffset = 0
        np_lines = []
        for j in range(n_j):
            aff_list = [
                int(i) - toffset for i in line.split()
            ]  # Taillard spec has machines id start at 1
            while len(aff_list) < n_m:
                aff_list.append(-1)
            np_lines.append(np.array(aff_list))
            line = next(f, "")
            if line == "":
                break
        affectations = np.stack(np_lines)

        if load_max_jobs == -1:
            load_max_jobs = n_j

        affectations = affectations[load_from_job : load_from_job + load_max_jobs]
        durations = durations[load_from_job : load_from_job + load_max_jobs]

        check_sanity(affectations, durations)

        return len(affectations), n_m, affectations, durations


def load_taillard_problem(problem_file, taillard_offset=True, deterministic=True):
    # http://jobshop.jjvh.nl/explanation.php#taillard_def

    if not deterministic:
        print("Loading problem with uncertainties, using extended taillard format")

    with open(problem_file, "r") as f:
        line = next(f)
        while line[0] == "#":
            line = next(f)

        # header
        header = line
        head_list = [int(i) for i in header.split()]
        assert len(head_list) == 2
        n_j = head_list[0]
        n_m = head_list[1]

        line = next(f)
        while line[0] == "#":
            line = next(f)

        # matrix of durations
        np_lines = []
        for j in range(n_j):
            dur_list = [float(i) for i in line.split()]
            np_lines.append(np.array(dur_list))
            line = next(f)
        durations = np.stack(np_lines)

        if deterministic:
            durations = np.expand_dims(durations, axis=2)
            durations = np.repeat(durations, 4, axis=2)
        else:
            mode_durations = durations

            while line[0] == "#":
                line = next(f)

            np_lines = []
            for j in range(n_j):
                dur_list = [float(i) for i in line.split()]
                np_lines.append(np.array(dur_list))
                line = next(f)
            min_durations = np.stack(np_lines)

            while line[0] == "#":
                line = next(f)

            np_lines = []
            for j in range(n_j):
                dur_list = [float(i) for i in line.split()]
                np_lines.append(np.array(dur_list))
                line = next(f)
            max_durations = np.stack(np_lines)

            real_durations = np.zeros((n_j, n_m)) - 1

            durations = np.stack(
                [real_durations, min_durations, max_durations, mode_durations], axis=2
            )

        while line[0] == "#":
            line = next(f)

        # matrix of affectations
        if taillard_offset:
            toffset = 1
        else:
            toffset = 0
        np_lines = []
        for j in range(n_j):
            aff_list = [
                int(i) - toffset for i in line.split()
            ]  # Taillard spec has machines id start at 1
            np_lines.append(np.array(aff_list))
            line = next(f, "")
            if line == "":
                break
        affectations = np.stack(np_lines)

        return n_j, n_m, affectations, durations


class PSPLoader:
    def __init__(self, deterministic=True, generate_bounds=-1):
        self.cleanup()
        self.deterministic = deterministic
        self.generate_bounds = -1

    def cleanup(self):
        self.f = None
        self.line = None
        self.sline = None
        self.fc = None
        self.fw = None

    def nextline(self):
        self.line = self.f.readline()
        self.sline = self.line.split()
        self.fc = self.line[0]
        self.fw = self.sline[0]

    def firstchar(self, c, stop=False):
        if self.fc != c:
            raise RuntimeError("bad first char" + c)
        if not stop:
            self.nextline()

    def firstword(self, w, stop=False):
        if self.fw != w:
            raise RuntimeError("bad first word " + w)
        if not stop:
            self.nextline()

    def word(self, n, w, stop=False):
        if self.sline[n] != w:
            raise RuntimeError("bad  word at pos " + str(n) + "  : " + w)
        if not stop:
            self.nextline()

    def load_directory(self, directory):
        files = glob.glob(directory + "/*")
        psps = [self.load_single(directory + "/" + f) for f in files]
        return psps

    def load_single(self, problem_file):

        self.f = open(problem_file, "r")
        self.nextline()

        if not self.deterministic:
            print("Loading problem with uncertainties, using customized format")
            if self.generate_bounds > 0:
                print(
                    "Generating random duration bounds of ", self.generate_bounds, " %"
                )
        job_info = []
        durations = []
        resources = []
        n_modes = 0

        self.firstchar("*")
        self.firstword("file")
        self.firstword("initial")
        self.firstchar("*")
        self.firstword("projects")
        self.firstword("jobs", True)
        n_jobs = int(self.sline[4])

        self.nextline()
        self.firstword("horizon")
        self.firstword("RESOURCES")
        self.word(1, "renewable", True)
        n_renewable_resources = int(self.sline[3])
        self.nextline()
        self.word(1, "nonrenewable", True)
        n_nonrenewable_resources = int(self.sline[3])
        self.nextline()
        self.word(1, "doubly", True)
        n_doubly_constrained_resources = int(self.sline[4])
        n_resources = (
            n_renewable_resources
            + n_nonrenewable_resources
            + n_doubly_constrained_resources
        )
        self.nextline()
        self.firstchar("*")
        self.firstword("PROJECT")
        self.firstword("pronr.")
        reldate = self.sline[2]
        duedate = self.sline[3]
        tardcost = self.sline[4]
        MPMtime = self.sline[5]
        self.nextline()
        self.firstchar("*")
        self.firstword("PRECEDENCE")
        self.firstword("jobnr.")
        for j in range(1, n_jobs + 1):
            n_modes += int(self.sline[1])
            job_info.append((int(self.sline[1]), [int(s) for s in self.sline[3:]]))
            self.nextline()
        self.firstchar("*")
        self.firstword("REQUESTS/DURATIONS:")
        self.firstword("jobnr.")
        self.firstchar("-")
        for j in range(1, n_jobs + 1):
            job_durations = []
            job_resources = []
            job_durations.append(int(self.sline[2]))
            job_resources.append([int(d) for d in self.sline[3:]])
            self.nextline()
            for m in range(1, job_info[j - 1][0]):
                job_durations.append(int(self.sline[1]))
                job_resources.append([int(d) for d in self.sline[2:]])
                self.nextline()
            durations.append(job_durations)
            resources.append(job_resources)
        self.firstchar("*")
        self.firstword("RESOURCEAVAILABILITIES:")
        self.nextline()
        resource_availabilities = [int(rl) for rl in self.sline]

        self.cleanup()

        return {
            "n_jobs": n_jobs,
            "n_modes": n_modes,
            "n_resources": n_resources,
            "n_renewable_resources": n_renewable_resources,
            "n_nonrenewable_resources": n_nonrenewable_resources,
            "n_doubly_constrained_resources": n_doubly_constrained_resources,
            "job_info": job_info,
            "durations": durations,
            "resources": resources,
            "resource_availabilities": resource_availabilities,
        }
