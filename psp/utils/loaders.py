#
# Wheatley
# Copyright (c) 2023 Jolibrain
# Authors:
#    Guillaume Infantes <guillaume.infantes@jolibrain.com>
#    Stéphanie Roussel <stephanie.roussel@onera.fr>
#
#
# This file is part of Wheatley.
#
# Wheatley is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Wheatley is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Wheatley. If not, see <https://www.gnu.org/licenses/>.
#
import numpy as np
import pathlib
from .rcpsp import Rcpsp
import glob
import re


class PSPLoader:
    def __init__(self, generate_bounds=None):
        self.cleanup()
        self.generate_bounds = generate_bounds

    def cleanup(self):
        self.f = None
        self.line = None
        self.sline = None
        self.fc = None
        self.fw = None

    def nextline(self):
        while True:
            self.line = self.f.readline()
            temp = re.split("\s|,|\(|\)|\n", self.line)
            self.sline = []
            for t in temp:
                if t != "":
                    self.sline.append(t)
            if len(self.sline) != 0:
                break
        self.fc = self.line[0]
        self.fw = self.sline[0]

    def firstchar(self, c, stop=False):
        if self.fc != c:
            raise RuntimeError("bad first char " + c + " found: '" + self.fc + "'")
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
        files = sorted(glob.glob(directory + "/*"))
        psps = [self.load_single(f) for f in files]
        print(f"loaded {len(psps)} files in {directory}")
        return psps

    def do_generate_bounds(self, mode):
        if self.generate_bounds is None:
            return mode, mode
        return mode * (1 - self.generate_bounds[0]), mode * (
            1 + self.generate_bounds[1]
        )

    def load_single(self, problem_file):
        # print("loading ", problem_file)
        suffix = pathlib.Path(problem_file).suffix
        if suffix in [".sm", ".mm"]:
            # return self.load_sm(problem_file)
            return self.load_single_rcpsp(problem_file)
        elif suffix == ".rcp":
            return self.load_rcp(problem_file)
        elif suffix == ".ag1":
            return self.load_ag1(problem_file)
        else:
            raise ValueError("unkown file format" + problem_file)

    def load_rcp(self, problem_file):
        self.f = open(problem_file, "r")
        self.nextline()
        job_info = []
        n_jobs = n_modes = int(self.sline[0])
        n_resources = n_renewable_resources = int(self.sline[1])
        n_nonrenewable_resources = 0
        n_doubly_constrained_resources = 0
        self.nextline()
        resource_availabilities = [int(rl) for rl in self.sline]
        max_resource_availability = max(resource_availabilities)
        max_resource_request = 0
        durations = [[], [], []]
        resources = []
        for j in range(n_jobs):
            self.nextline()
            job_info.append((1, [int(s) for s in self.sline[5:]]))

            job_durations = [[], [], []]
            job_resources = []
            job_durations[0].append(int(self.sline[0]))
            if len(self.sline) == (n_resources + 7):
                job_durations[1].append(int(self.sline[1]))
                job_durations[2].append(int(self.sline[2]))
                startr = 3
            else:
                dmin, dmax = self.do_generate_bounds(job_durations[0][-1])
                job_durations[1].append(dmin)
                job_durations[2].append(dmax)
                startr = 1
            req = [int(d) for d in self.sline[startr : startr + n_resources]]
            if max(req) > max_resource_request:
                max_resource_request = max(req)
            job_resources.append(req)
            for i in range(3):
                durations[i].append(job_durations[i])
            resources.append(job_resources)
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
            "resource_availability": resource_availabilities,
            "max_resource_availability": max_resource_availability,
            "max_resource_request": max_resource_request,
        }

    def load_ag1(self, problem_file):
        self.f = open(problem_file, "r")
        self.nextline()
        for i in range(18):
            self.firstchar("#")
        n_renew_r = int(self.sline[2])
        print("n_renew_r", n_renew_r)
        self.nextline()
        n_nonrenew_r = int(self.sline[2])
        print("n_nonrenew_r", n_nonrenew_r)
        for i in range(n_renew_r + n_nonrenew_r + 1):
            self.nextline()
        resource_availabilities = [int(self.sline[2])]
        for i in range(n_renew_r + n_nonrenew_r - 1):
            self.nextline()
            resource_availabilities.append(int(self.sline[0]))
        self.nextline()
        if self.sline[3] == "1":
            use_index_from_zero = False
            print("index starting at 1")
        else:
            use_index_from_zero = True
            print("index starting at 0")
        self.nextline()
        n_jobs = 1
        while self.sline[0] != "N_modes_per_job":
            n_jobs += 1
            self.nextline()
        print("n_jobs", n_jobs)

        n_modes_per_job = [int(self.sline[2])]
        self.nextline()
        for i in range(n_jobs - 1):
            n_modes_per_job.append(int(self.sline[0]))
            self.nextline()

        suc = int(self.sline[2])
        if suc == -1:
            successors = [[]]
        else:
            if use_index_from_zero:
                successors = [[suc + 1]]
            else:
                successors = [[suc]]
        self.nextline()

        for i in range(n_jobs - 1):
            suc = int(self.sline[0])
            if suc == -1:
                successors.append([])
            else:
                if use_index_from_zero:
                    successors.append([suc + 1])
                else:
                    successors.append([suc])
            self.nextline()
        due_dates = [int(self.sline[2])]
        self.nextline()
        for i in range(n_jobs - 1):
            due_dates.append(self.sline[0])
            self.nextline()

        resource_cons = []
        resource_cons.append(
            [[int(c) for c in self.sline[2 : 2 + n_renew_r + n_nonrenew_r]]]
        )
        self.nextline()
        for i in range(n_jobs - 1):
            resource_cons.append(
                [[int(c) for c in self.sline[0 : n_renew_r + n_nonrenew_r]]]
            )
            self.nextline()
        durations = [[], [], []]
        for j in range(3):
            durations[j].append([float(self.sline[2 + j])])
        self.nextline()
        for i in range(n_jobs - 2):
            for j in range(3):
                durations[j].append([float(self.sline[j])])
            self.nextline()

        for j in range(3):
            durations[j].append([float(self.sline[j])])

        return Rcpsp(
            n_jobs=n_jobs,
            n_modes_per_job=n_modes_per_job,
            successors=successors,
            durations=durations,
            resource_cons=resource_cons,
            resource_availabilities=resource_availabilities,
            n_renewable_resources=n_renew_r,
            n_nonrenewable_resources=n_nonrenew_r,
            n_doubly_constrained_resources=0,
            use_index_from_zero=use_index_from_zero,
            due_dates=due_dates,
        )

    def load_sm(self, problem_file):
        self.f = open(problem_file, "r")
        self.nextline()

        if self.generate_bounds is not None:
            print(
                "If not present, generating random duration bounds of ",
                self.generate_bounds,
                " %",
            )
        job_info = []
        durations = [[], [], []]
        resources = []
        n_modes = 0
        max_resource_availability = 0
        max_resource_request = 0

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
            job_durations = [[], [], []]
            job_resources = []
            job_durations[0].append(int(self.sline[2]))
            if len(self.sline) == (n_resources + 5):
                job_durations[1].append(int(self.sline[3]))
                job_durations[2].append(int(self.sline[4]))
                startr = 5
            else:
                dmin, dmax = self.do_generate_bounds(job_durations[0][-1])
                job_durations[1].append(dmin)
                job_durations[2].append(dmax)
                startr = 3
            req = [int(d) for d in self.sline[startr:]]
            if max(req) > max_resource_request:
                max_resource_request = max(req)
            job_resources.append(req)
            self.nextline()
            for m in range(1, job_info[j - 1][0]):
                job_durations[0].append(int(self.sline[1]))
                if len(self.sline) == (n_resources + 4):
                    job_durations[1].append(int(self.sline[3]))
                    job_durations[2].append(int(self.sline[4]))
                    startr = 4
                else:
                    dmin, dmax = self.do_generate_bounds(job_durations[0][-1])
                    job_durations[1].append(dmin)
                    job_durations[2].append(dmax)
                    startr = 2
                job_resources.append([int(d) for d in self.sline[startr:]])
                self.nextline()
            for i in range(3):
                durations[i].append(job_durations[i])

            resources.append(job_resources)
        self.firstchar("*")
        self.firstword("RESOURCEAVAILABILITIES:")
        self.nextline()
        resource_availabilities = [int(rl) for rl in self.sline]
        if max(resource_availabilities) > max_resource_availability:
            max_resource_availability = max(resource_availabilities)

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
            "resource_availability": resource_availabilities,
            "max_resource_availability": max_resource_availability,
            "max_resource_request": max_resource_request,
        }

    def load_single_rcpsp(self, problem_file):
        self.f = open(problem_file, "r")
        self.nextline()

        if self.generate_bounds is not None:
            print(
                "If not present, generating random duration bounds of ",
                self.generate_bounds,
                " %",
            )
        job_info = []
        durations = []
        resources = []
        n_modes = 0
        max_resource_availability = 0
        max_resource_request = 0

        # Number of jobs in the graph
        n_jobs = 0
        # Number of modes for each job
        n_modes_per_job = []
        # Successors for each job. Successors are given by a list
        successors = []
        job_ids = []
        # Capacity of each resource
        resource_availabilities = []
        # Number of renewable resources
        n_renewable_resources = 0
        # Number of non renewable resources
        n_nonrenewable_resources = 0
        # Number of doubly constrained resources
        n_doubly_constrained_resources = 0
        # Durations for each job and for each mode. Durations are expressed through an array [MIN, MAX, MOD]
        durations = [[], [], []]
        # Consumption for each job and for each mode of jobs
        resources_cons = []

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
        for j in range(n_jobs):
            job_ids.append(self.sline[0])
            n_modes += int(self.sline[1])
            n_modes_per_job.append(int(self.sline[1]))
            successors.append([s for s in self.sline[3:]])
            self.nextline()
        self.firstchar("*")
        self.firstword("REQUESTS/DURATIONS:")
        self.firstword("jobnr.")
        self.firstchar("-")
        for j in range(n_jobs):
            job_durations = [[], [], []]
            job_resources = []
            job_durations[0].append(int(self.sline[2]))
            if len(self.sline) == (n_resources + 5):
                job_durations[1].append(int(self.sline[3]))
                job_durations[2].append(int(self.sline[4]))
                startr = 5
            else:
                dmin, dmax = self.do_generate_bounds(job_durations[0][-1])
                job_durations[1].append(dmin)
                job_durations[2].append(dmax)
                startr = 3

            req = [int(d) for d in self.sline[startr:]]
            job_resources.append(req)
            self.nextline()
            # Starts at 1 so that the code is only executed for jobs that have more than 1 mode
            for m in range(1, n_modes_per_job[j]):
                job_durations[0].append(int(self.sline[1]))
                if len(self.sline) == (n_resources + 4):
                    job_durations[1].append(int(self.sline[3]))
                    job_durations[2].append(int(self.sline[4]))
                    startr = 4
                else:
                    dmin, dmax = self.do_generate_bounds(job_durations[0][-1])
                    job_durations[1].append(dmin)
                    job_durations[2].append(dmax)
                    startr = 2
                job_resources.append([int(d) for d in self.sline[startr:]])
                self.nextline()
            for i in range(3):
                durations[i].append(job_durations[i])
            resources_cons.append(job_resources)
        self.firstchar("*")
        self.firstword("RESOURCEAVAILABILITIES:")
        self.nextline()
        resource_availabilities = [int(rl) for rl in self.sline]

        self.cleanup()

        return Rcpsp(
            pb_id=problem_file,
            n_jobs=n_jobs,
            job_ids=job_ids,
            n_modes_per_job=n_modes_per_job,
            successors=successors,
            durations=durations,
            resource_cons=resources_cons,
            resource_availabilities=resource_availabilities,
            n_renewable_resources=n_renewable_resources,
            n_nonrenewable_resources=n_nonrenewable_resources,
            n_doubly_constrained_resources=n_doubly_constrained_resources,
        )
